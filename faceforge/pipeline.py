"""
FaceForge pipeline — orchestrates the full face reconstruction workflow.

High-level flow
---------------
1. Canonical preprocessing  (CanonicalPreprocessor) — detect + align per image
2. MICA encoding             (MICAAdapter)           — shape code per aligned image
3. Multi-image aggregation   (MultiImageAggregator)  — robust shape estimate
4. ShapeRefiner              (ShapeRefiner)          — differentiable refinement
5. Export                                            — mesh / params / renders
"""

from dataclasses import dataclass
from typing import Union, List, Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf


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
    Full reconstruction pipeline.
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
        from faceforge.preprocess.stage import CanonicalPreprocessor
        from faceforge.encoder.mica_adapter import MICAAdapter

        dev_str = str(getattr(config, "device", "auto"))
        self.device = get_device() if dev_str == "auto" else torch.device(dev_str)

        logger.info(f"[Pipeline] Initializing on device: {self.device}")

        # Legacy encoder (kept for backward compatibility)
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

        # NEW: canonical preprocessing stage and thin MICA adapter
        self.preprocessor = CanonicalPreprocessor(config, self.device)
        self.mica = MICAAdapter(config, self.device)

        self.config = config

    def run(
        self,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        output_dir: str = "output",
        subject_id: str = "subject",
    ) -> PipelineResult:
        """Full pipeline: preprocess → MICA encode → aggregate → refine → export."""
        from pathlib import Path
        from faceforge.utils.image import load_image
        from faceforge.utils.mesh_io import save_mesh
        from faceforge.utils.artifacts import ArtifactWriter

        # Normalize to list
        if not isinstance(images, list):
            images = [images]

        # Load all images
        loaded = [load_image(img) for img in images]

        logger.info(
            f"[Pipeline] Processing {len(loaded)} image(s) for subject '{subject_id}'"
        )

        # Artifact writer (no-op when save_intermediates=False)
        save_intermediates = bool(
            getattr(self.config.output, "save_intermediates", False)
        )
        artifacts = ArtifactWriter(output_dir, subject_id, enabled=save_intermediates)
        artifacts.ensure_stage_dirs()

        # Stage 1: Canonical preprocessing + MICA encoding per image
        all_shape_codes = []
        for i, img in enumerate(loaded):
            artifacts.save_input(img, name=f"input_{i:03d}.png")
            pre_result = self.preprocessor.run(img)
            artifacts.save_aligned(pre_result.aligned_image, name=f"aligned_{i:03d}.png")
            artifacts.save_landmark_preview(pre_result.preview_image, name=f"landmarks_{i:03d}.png")
            mica_result = self.mica.run(pre_result)
            artifacts.save_mica_preview(
                pre_result.aligned_image, name=f"mica_preview_{i:03d}.png"
            )
            all_shape_codes.append(mica_result.shape_code)

        # Stage 2: Multi-image aggregation — pass pre-computed (N, 300) tensor
        stacked_shapes = torch.cat(all_shape_codes, dim=0)  # (N, 300)
        agg_result = self.aggregator.aggregate(stacked_shapes)
        shape_init = agg_result.shape_params  # (1, 300)
        confidence = agg_result.confidence

        logger.info(
            f"[Pipeline] Confidence: {confidence:.3f}, "
            f"valid images: {agg_result.n_valid_images}"
        )

        # Stage 3: Optional refinement
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
            loss_final = (
                refine_result.loss_history[-1] if refine_result.loss_history else 0.0
            )
        else:
            shape_final = shape_init

        # Stage 4: Generate final mesh via FLAME
        expr = torch.zeros(1, 50, device=self.device)
        pose = torch.zeros(1, 6, device=self.device)
        flame_out = self.flame(shape_final, expr, pose)

        # Stage 5: Export
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
