"""
FaceForge pipeline — orchestrates the full face reconstruction workflow.
"""

from dataclasses import dataclass
from typing import Union, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from loguru import logger


@dataclass
class PipelineResult:
    mesh_path:    str          # saved .ply path (empty string if save_mesh=False)
    render_path:  str          # saved render preview path (empty if save_render=False)
    params_path:  str          # saved .npy params path (empty if save_params=False)
    shape_params: np.ndarray   # (300,) FLAME shape coefficients
    confidence:   float        # multi-image consistency score (1.0 for single image)
    loss_final:   float        # final loss value (0.0 if refiner disabled)


class FaceForgePipeline:
    """
    Full pipeline:
    1. Image preprocessing (face detection + alignment)
    2. MICA encoding (single or multi-image median aggregation)
    3. ShapeRefiner optimization (optional, default on)
    4. Output: mesh / render preview / params
    """

    def __init__(self, config: OmegaConf):
        from faceforge.utils.device import get_device
        from faceforge.encoder.mica_encoder import MICAEncoder
        from faceforge.encoder.multi_image import MultiImageAggregator
        from faceforge.model.flame import FLAMELayer
        from faceforge.model.renderer import DifferentiableRenderer
        from faceforge.optimizer.refiner import ShapeRefiner
        from faceforge.optimizer.losses import LossWeights
        from faceforge.utils.landmarks import LandmarkDetector
        from faceforge.utils.image import FaceDetector

        dev_str = str(getattr(config, "device", "auto"))
        self.device = get_device() if dev_str == "auto" else torch.device(dev_str)

        logger.info(f"[Pipeline] Initializing on device: {self.device}")
        self.encoder = MICAEncoder(
            config.paths.mica_weights, config.paths.flame_model, self.device,
            insightface_name=config.encoder.insightface_name,
        )
        self.aggregator = MultiImageAggregator(
            self.encoder, config.aggregator.strategy
        )
        self.flame = FLAMELayer(config.paths.flame_model, self.device)
        self.renderer = DifferentiableRenderer(config.refiner.render_size, self.device)

        losses_cfg = config.refiner.losses
        lw = LossWeights(
            landmark=losses_cfg.landmark,
            photometric=losses_cfg.photometric,
            identity=losses_cfg.identity,
            contour=losses_cfg.contour,
            region=losses_cfg.region,
            regularize=losses_cfg.regularize,
        )
        self.refiner = ShapeRefiner(
            flame=self.flame,
            renderer=self.renderer,
            lmk_detector=LandmarkDetector(self.device),
            face_detector=FaceDetector(model_name=config.encoder.insightface_name),
            loss_weights=lw,
            device=self.device,
        )
        self.config = config

    def run(
        self,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        output_dir: str = "output",
        subject_id: str = "subject",
    ) -> PipelineResult:
        """
        Full pipeline flow:
        1. Normalize images to list
        2. Multi-image aggregation -> shape_init
        3. If refiner.enabled: ShapeRefiner.refine()
        4. FLAMELayer.forward() to get final mesh
        5. Save mesh / render / params to output_dir
        6. Return PipelineResult
        """
        from pathlib import Path
        from faceforge.utils.image import load_image
        from faceforge.utils.mesh_io import save_mesh

        # Normalize to list
        if not isinstance(images, list):
            images = [images]

        # Load all images
        loaded = [load_image(img) for img in images]

        logger.info(f"[Pipeline] Encoding {len(loaded)} image(s)...")

        # Aggregate shape params
        agg_result = self.aggregator.aggregate(loaded)
        shape_init = agg_result.shape_params  # (1, 300)
        confidence = agg_result.confidence

        logger.info(f"[Pipeline] Confidence: {confidence:.3f}, valid images: {agg_result.n_valid_images}")

        # Refinement
        loss_final = 0.0
        if self.config.refiner.enabled:
            # Use first image as primary reference, resized to render_size
            import cv2 as _cv2
            render_size = self.config.refiner.render_size
            img_resized = _cv2.resize(loaded[0], (render_size, render_size))
            img_tensor = torch.from_numpy(
                img_resized.astype(np.float32) / 255.0
            ).unsqueeze(0).to(self.device)  # (1, render_size, render_size, 3)

            cam_params = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

            refine_result = self.refiner.refine(
                shape_init=shape_init,
                image=img_tensor,
                cam_params=cam_params,
                n_steps=self.config.refiner.n_steps,
                lr=self.config.refiner.lr,
                verbose=True,
            )
            shape_final = refine_result.shape_params
            loss_final = refine_result.loss_history[-1] if refine_result.loss_history else 0.0
        else:
            shape_final = shape_init

        # Generate final mesh
        expr = torch.zeros(1, 50, device=self.device)
        pose = torch.zeros(1, 6, device=self.device)
        flame_out = self.flame(shape_final, expr, pose)

        # Save outputs
        out_path = Path(output_dir)
        mesh_dir = out_path / "meshes"
        render_dir = out_path / "renders"
        params_dir = out_path / "params"
        for d in [mesh_dir, render_dir, params_dir]:
            d.mkdir(parents=True, exist_ok=True)

        mesh_path = ""
        render_path = ""
        params_path = ""

        if self.config.output.save_mesh:
            fmt = self.config.output.mesh_format
            mesh_path = str(mesh_dir / f"{subject_id}.{fmt}")
            verts = flame_out.vertices[0].detach().cpu().numpy()
            faces = flame_out.faces.detach().cpu().numpy()
            save_mesh(mesh_path, verts, faces, format=fmt)
            logger.info(f"[Pipeline] Mesh saved: {mesh_path}")

        if self.config.output.save_params:
            params_path = str(params_dir / f"{subject_id}_shape.npy")
            np.save(params_path, shape_final.detach().cpu().numpy())
            logger.info(f"[Pipeline] Params saved: {params_path}")

        return PipelineResult(
            mesh_path=mesh_path,
            render_path=render_path,
            params_path=params_path,
            shape_params=shape_final.detach().cpu().numpy().squeeze(),
            confidence=confidence,
            loss_final=loss_final,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "FaceForgePipeline":
        base = OmegaConf.load("config/default.yaml")
        override = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(base, override)
        return cls(cfg)
