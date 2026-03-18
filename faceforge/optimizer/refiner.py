"""
Iterative FLAME parameter refiner: gradient-based optimization loop.
"""

import torch
import torch.optim as optim
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


@dataclass
class RefineResult:
    shape_params:  torch.Tensor    # (1, 300) optimized shape coefficients
    loss_history:  List[float]     # total loss at each step
    flame_output:  object          # FLAMEOutput from final step
    n_steps_done:  int


class ShapeRefiner:
    """
    MICA initialization + differentiable rendering optimization.

    Optimizes: shape_params (300-dim)
    Fixed: expression (zeros), pose (zeros), camera params

    Key design:
    - shape_regularizer keeps optimization near MICA init (weight=0.1)
    - CosineAnnealingLR prevents oscillation
    - IdentityLoss and RegionWeightedLoss are optional
    """

    def __init__(
        self,
        flame,
        renderer,
        lmk_detector,
        face_detector,
        loss_weights,
        identity_loss=None,
        region_loss=None,
        device=None,
    ):
        from faceforge.utils.device import get_device
        self.flame = flame
        self.renderer = renderer
        self.lmk_detector = lmk_detector
        self.face_detector = face_detector
        self.lw = loss_weights
        self.identity_loss = identity_loss
        self.region_loss = region_loss
        self.device = device or get_device()

    def refine(
        self,
        shape_init,
        image,
        cam_params,
        n_steps=100,
        lr=1e-3,
        verbose=False,
    ):
        """
        Optimization loop:
        1. shape = shape_init.clone().requires_grad_(True)
        2. Adam + CosineAnnealingLR
        3. Each step: forward FLAME -> render -> compute all losses -> backward
        4. Record loss_history
        5. Return RefineResult with best shape (lowest loss)
        """
        from faceforge.optimizer.losses import (
            landmark_loss, photometric_loss, contour_loss,
            shape_regularizer, extract_face_contour,
        )

        shape_init = shape_init.to(self.device)
        image = image.to(self.device)
        cam_params = cam_params.to(self.device)

        # Pre-compute targets (don't change during optimization)
        # Detect landmarks from input image
        img_np = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        try:
            target_lmk = torch.from_numpy(
                self.lmk_detector.detect(img_np, normalize=True)
            ).unsqueeze(0).to(self.device)  # (1, 68, 2)
        except Exception as e:
            logger.warning(f"[ShapeRefiner] Landmark detection failed: {e}. Using zeros.")
            target_lmk = torch.zeros(1, 68, 2, device=self.device)

        # Face mask from detector
        try:
            faces = self.face_detector._app.get(img_np)
            if faces:
                face_mask = self._build_face_mask(faces[0], image.shape[1:3])
            else:
                face_mask = torch.ones(1, image.shape[1], image.shape[2], device=self.device)
        except Exception:
            face_mask = torch.ones(1, image.shape[1], image.shape[2], device=self.device)

        face_contour = extract_face_contour(face_mask)  # (1, 100, 2)

        # Setup optimization
        shape = shape_init.clone().requires_grad_(True)
        expr = torch.zeros(1, 50, device=self.device)
        pose = torch.zeros(1, 6, device=self.device)

        optimizer = optim.Adam([shape], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
        cameras = self.renderer.build_cameras(cam_params, self.renderer.image_size)

        loss_history = []
        best_loss = float("inf")
        best_shape = shape_init.clone()
        ref_vertices = None

        for step in range(n_steps):
            optimizer.zero_grad()

            flame_out = self.flame(shape, expr, pose)
            render_out = self.renderer.render(flame_out.vertices, flame_out.faces, cameras)

            # Initialize ref_vertices on first step
            if ref_vertices is None:
                ref_vertices = flame_out.vertices.detach().clone()

            loss = torch.tensor(0.0, device=self.device)

            # 1. Landmark loss
            if self.lw.landmark > 0:
                proj_lmk = self.renderer.project_points(flame_out.landmarks3d, cameras)
                # Normalize projected landmarks to [0,1] range
                proj_lmk_norm = (proj_lmk / self.renderer.image_size).clamp(0, 1)
                loss = loss + self.lw.landmark * landmark_loss(proj_lmk_norm, target_lmk)

            # 2. Photometric loss
            if self.lw.photometric > 0:
                skin_mask = self.renderer.extract_face_mask(render_out).float()
                rendered_rgb = render_out.image[..., :3]
                loss = loss + self.lw.photometric * photometric_loss(rendered_rgb, image, skin_mask)

            # 3. Identity loss (optional)
            if self.identity_loss is not None and self.lw.identity > 0:
                loss = loss + self.lw.identity * self.identity_loss(
                    render_out.image[..., :3], image
                )

            # 4. Contour loss
            if self.lw.contour > 0:
                contour_verts = self.flame.get_contour_vertices(flame_out.vertices)
                contour_2d = self.renderer.project_points(contour_verts, cameras)
                contour_2d_norm = (contour_2d / self.renderer.image_size).clamp(0, 1) * 2 - 1
                loss = loss + self.lw.contour * contour_loss(contour_2d_norm, face_contour)

            # 5. Region loss (optional)
            if self.region_loss is not None and self.lw.region > 0:
                loss = loss + self.lw.region * self.region_loss(
                    flame_out.vertices, ref_vertices
                )

            # 6. Shape regularizer
            if self.lw.regularize > 0:
                loss = loss + self.lw.regularize * shape_regularizer(shape, shape_init)

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_shape = shape.detach().clone()

            if verbose and step % 10 == 0:
                logger.info(f"  step {step:3d} | loss {loss_val:.4f}")

        return RefineResult(
            shape_params=best_shape,
            loss_history=loss_history,
            flame_output=flame_out,
            n_steps_done=n_steps,
        )

    def _build_face_mask(self, face, image_hw):
        """Build a binary face mask from InsightFace detection bbox."""
        import numpy as np
        H, W = image_hw
        mask = torch.zeros(1, H, W, device=self.device)
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(W, bbox[2]), min(H, bbox[3])
        mask[0, y1:y2, x1:x2] = 1.0
        return mask
