"""
Loss functions for the iterative FLAME parameter refiner:
landmark, photometric, identity, contour, region, and regularization losses.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LossWeights:
    landmark:    float = 1.0
    photometric: float = 0.5
    identity:    float = 0.3
    contour:     float = 0.5
    region:      float = 0.8
    regularize:  float = 0.1


# ── 1. Landmark loss ─────────────────────────────────────────────────
def landmark_loss(
    projected_lmk: torch.Tensor,   # (B, 68, 2) projected 2D coords (normalized [-1,1])
    target_lmk:    torch.Tensor,   # (B, 68, 2) detected 2D coords (normalized)
) -> torch.Tensor:
    """Mean squared error (MSE) between projected and detected 2D landmark coordinates."""
    return F.mse_loss(projected_lmk, target_lmk)


# ── 2. Photometric loss ──────────────────────────────────────────────
def photometric_loss(
    rendered:   torch.Tensor,  # (B, H, W, 3) rendered image [0,1]
    target:     torch.Tensor,  # (B, H, W, 3) input image [0,1]
    skin_mask:  torch.Tensor,  # (B, H, W)    skin region mask (bool or float)
) -> torch.Tensor:
    """
    L1 pixel difference, computed only within skin_mask region.
    Excludes eyes, mouth interior, hair to avoid incorrect gradients.
    """
    if skin_mask.sum() == 0:
        logger.warning("[photometric_loss] skin_mask is empty — loss will be 0. Check mask computation.")
        return torch.tensor(0.0, device=rendered.device, requires_grad=rendered.requires_grad)
    mask = skin_mask.unsqueeze(-1).float()
    if rendered.shape[-1] == 4:
        rendered = rendered[..., :3]  # drop alpha if RGBA
    C = rendered.shape[-1]
    diff = (rendered - target).abs() * mask
    # Denominator counts masked pixels times number of channels
    denom = (mask.sum() * C).clamp(min=1.0)
    return diff.sum() / denom


# ── 3. Identity loss (ArcFace) ───────────────────────────────────────
class IdentityLoss(nn.Module):
    """
    Maximize cosine similarity between ArcFace embeddings of rendered and target.
    Loss = 1 - cosine_similarity(feat_rendered, feat_target)
    """
    def __init__(self, arcface_model):
        super().__init__()
        self.arcface = arcface_model

    def forward(
        self,
        rendered: torch.Tensor,  # (B, H, W, 3) [0,1]
        target:   torch.Tensor,  # (B, H, W, 3) [0,1]
    ) -> torch.Tensor:
        feat_rendered = self.arcface(self._preprocess(rendered))
        feat_target   = self.arcface(self._preprocess(target))
        sim = F.cosine_similarity(feat_rendered, feat_target, dim=1)
        return (1.0 - sim).mean()

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Resize to 112x112 and normalize for ArcFace input."""
        # img: (B, H, W, 3) [0,1] -> need (B, 3, 112, 112) [-1,1]
        # BHWC -> BCHW
        x = img.permute(0, 3, 1, 2)  # (B, 3, H, W)
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0  # normalize to [-1, 1]
        return x


# ── 4. Contour loss (HRN-style) ──────────────────────────────────────
def contour_loss(
    contour_verts_2d: torch.Tensor,  # (B, N, 2) projected contour vertices
    face_contour_2d:  torch.Tensor,  # (B, M, 2) face edge points from mask
) -> torch.Tensor:
    """
    Bidirectional Chamfer distance between mesh silhouette and face contour.
    Direction 1: for each contour_vert, find nearest face_contour point
    Direction 2: for each face_contour point, find nearest contour_vert
    Returns mean of both directions.
    """
    # p: (B, N, 2), q: (B, M, 2)
    p = contour_verts_2d.unsqueeze(2)   # (B, N, 1, 2)
    q = face_contour_2d.unsqueeze(1)    # (B, 1, M, 2)
    dist = ((p - q) ** 2).sum(-1)       # (B, N, M)
    p2q = dist.min(dim=2).values.mean() # nearest in q for each p
    q2p = dist.min(dim=1).values.mean() # nearest in p for each q
    return (p2q + q2p) / 2.0


def extract_face_contour(
    face_mask: torch.Tensor,   # (B, H, W) bool or float
    n_points:  int = 100,
) -> torch.Tensor:             # (B, n_points, 2) coords normalized to [-1, 1]
    """
    Extract ~n_points contour points from face segmentation mask using Canny edges.
    Uniformly subsamples the edge points to n_points.

    Note: Uses OpenCV Canny (non-differentiable). Returns a detached tensor.
    Gradients flow through contour_loss, not through this function.
    """
    import cv2
    import numpy as np
    B, H, W = face_mask.shape
    device = face_mask.device
    result = []
    for b in range(B):
        mask_np = (face_mask[b].detach().cpu().float().numpy() * 255).astype(np.uint8)
        edges = cv2.Canny(mask_np, 50, 150)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            pts = np.zeros((n_points, 2), dtype=np.float32)
        else:
            idx = np.linspace(0, len(xs) - 1, n_points, dtype=int)
            # Normalize to [-1, 1]
            pts = np.stack([
                xs[idx].astype(np.float32) / W * 2 - 1,
                ys[idx].astype(np.float32) / H * 2 - 1,
            ], axis=1)
        result.append(torch.from_numpy(pts))
    return torch.stack(result).to(device)


# ── 5. Region-weighted loss (HiFace-style) ───────────────────────────
REGION_WEIGHTS = {
    "nose":     3.0,
    "eyes":     2.5,
    "mouth":    2.0,
    "jaw":      1.5,
    "cheeks":   1.0,
    "forehead": 0.5,
}


class RegionWeightedLoss(nn.Module):
    """
    Per-region vertex L2 loss with facial region weighting.
    Loads FLAME vertex masks from FLAME_masks.pkl if available.
    Falls back to uniform vertex loss if masks not available.
    """

    def __init__(self, flame_masks_path: str):
        super().__init__()
        self._region_ids = {}
        self._region_tensors = {}
        self._load_masks(flame_masks_path)

    def _load_masks(self, path: str) -> None:
        """Load per-region vertex indices from FLAME_masks.pkl."""
        from pathlib import Path
        import pickle
        if not Path(path).exists():
            logger.warning(f"[RegionWeightedLoss] FLAME masks not found: {path}. Using uniform vertex loss.")
            return
        with open(path, "rb") as f:
            masks = pickle.load(f, encoding="latin1")
        # FLAME_masks.pkl keys are region names, values are arrays of vertex indices
        region_map = {
            "nose": ["nose"],
            "eyes": ["left_eye_region", "right_eye_region", "left_eyeball", "right_eyeball"],
            "mouth": ["lips", "mouth"],
            "jaw": ["jaw"],
            "cheeks": ["cheeks", "left_cheek", "right_cheek"],
            "forehead": ["forehead"],
        }
        for region, keys in region_map.items():
            indices = []
            for key in keys:
                if key in masks:
                    v = masks[key]
                    if hasattr(v, "tolist"):
                        indices.extend(v.tolist())
                    else:
                        indices.extend(list(v))
            if indices:
                self._region_ids[region] = list(set(indices))
        self._region_tensors = {
            region: torch.tensor(ids, dtype=torch.long)
            for region, ids in self._region_ids.items()
        }

    def forward(
        self,
        pred_vertices: torch.Tensor,  # (B, 5023, 3) predicted vertices
        ref_vertices:  torch.Tensor,  # (B, 5023, 3) reference vertices (MICA init)
    ) -> torch.Tensor:
        """Weighted region vertex L2 distance."""
        if not self._region_ids:
            # No masks loaded — fall back to uniform vertex L2
            return F.mse_loss(pred_vertices, ref_vertices)

        total = torch.zeros(1, device=pred_vertices.device)
        total_weight = 0.0
        for region, weight in REGION_WEIGHTS.items():
            ids = self._region_ids.get(region)
            if ids is None:
                continue
            idx = self._region_tensors[region].to(pred_vertices.device)
            diff = (pred_vertices[:, idx] - ref_vertices[:, idx]).pow(2).mean()
            total = total + weight * diff
            total_weight += weight

        if total_weight == 0:
            return F.mse_loss(pred_vertices, ref_vertices)
        return total / total_weight


# ── 6. Shape regularizer ─────────────────────────────────────────────
def shape_regularizer(
    shape_params:  torch.Tensor,  # (B, 300) current optimized coefficients
    shape_init:    torch.Tensor,  # (B, 300) MICA initialization
    lambda_prior:  float = 1.0,
) -> torch.Tensor:
    """
    Two terms:
    1. L2(shape, shape_init): don't drift far from MICA initialization
    2. PCA prior: shape.pow(2).mean() — encourages coefficients near 0
    Returns: anchor_loss + lambda_prior * prior_loss
    """
    anchor_loss = F.mse_loss(shape_params, shape_init)
    prior_loss  = shape_params.pow(2).mean()
    return anchor_loss + lambda_prior * prior_loss
