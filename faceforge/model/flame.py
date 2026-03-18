"""
FLAME 2020 parametric face model wrapper.
Differentiable linear blend skinning layer with shape, expression, and pose parameters.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Landmark indices (68-point dlib order for FLAME 2020)
# ---------------------------------------------------------------------------

FLAME_LANDMARK_INDICES = [
    # Jaw line (0-16): 17 points
    2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598,
    3637, 3587, 3582, 3580, 3579, 3575, 3572,
    # Right eyebrow (17-21)
    2441, 2540, 2439, 2407, 2216,
    # Left eyebrow (22-26)
    2782, 2681, 2682, 2713, 2891,
    # Nose bridge (27-30)
    3514, 3538, 3541, 3543,
    # Nose tip (31-35)
    1781, 1550, 1693, 1709, 1808,
    # Right eye (36-41)
    2215, 2393, 2444, 2497, 2416, 2349,
    # Left eye (42-47)
    2780, 2968, 2886, 2938, 2967, 2832,
    # Outer lip (48-59)
    1572, 1681, 1710, 1736, 1745, 1843, 1915, 1859, 1825, 1801, 1779, 1672,
    # Inner lip (60-67)
    1573, 1682, 1711, 1738, 1744, 1842, 1912, 1857,
]

FLAME_CONTOUR_INDICES = FLAME_LANDMARK_INDICES[:17]  # jaw line


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FLAMEOutput:
    vertices:    torch.Tensor  # (B, 5023, 3)
    faces:       torch.Tensor  # (9976, 3) — constant, doesn't change per batch
    landmarks2d: torch.Tensor  # (B, 68, 2) projected to [-1, 1]
    landmarks3d: torch.Tensor  # (B, 68, 3)


# ---------------------------------------------------------------------------
# FLAMELayer
# ---------------------------------------------------------------------------

class FLAMELayer(nn.Module):
    """
    FLAME differentiable layer. Loads generic_model.pkl, implements linear blend deformation.
    Fully differentiable — supports gradient backprop through shape/expression params.
    """

    NUM_VERTS = 5023
    NUM_FACES = 9976
    SHAPE_DIM = 300
    EXPR_DIM  = 50
    POSE_DIM  = 6  # global 3 + jaw 3

    def __init__(self, flame_model_path: str, device=None):
        super().__init__()

        if Path(flame_model_path).exists():
            self._load_from_file(flame_model_path)
        else:
            logger.warning(
                f"[FLAME] Model not found: {flame_model_path}. Using zero initialization."
            )
            self._init_zeros()

        # Register landmark indices as buffers
        self.register_buffer(
            "landmark_indices",
            torch.tensor(FLAME_LANDMARK_INDICES, dtype=torch.long),
        )
        self.register_buffer(
            "contour_indices",
            torch.tensor(FLAME_CONTOUR_INDICES, dtype=torch.long),
        )

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_from_file(self, path: str) -> None:
        """Load FLAME 2020 pkl and register buffers."""
        with open(path, "rb") as f:
            flame_data = pickle.load(f, encoding="latin1")

        V = self.NUM_VERTS

        # v_template  (V, 3)
        v_template = np.array(flame_data["v_template"], dtype=np.float32)
        self.register_buffer("v_template", torch.from_numpy(v_template))

        # shapedirs  (V, 3, n_basis) — may be sparse; densify first
        shapedirs = flame_data["shapedirs"]
        if hasattr(shapedirs, "toarray"):  # scipy sparse
            shapedirs = shapedirs.toarray()
        shapedirs = np.array(shapedirs, dtype=np.float32)

        if shapedirs.shape[-1] >= 400:
            shape_dirs = shapedirs[..., :300]          # (V, 3, 300)
            expr_dirs  = shapedirs[..., 300:350]       # (V, 3, 50)
        else:
            shape_dirs = shapedirs                     # (V, 3, <=300)
            raw_expr = flame_data.get("exprdirs", np.zeros((V, 3, 50), dtype=np.float32))
            if hasattr(raw_expr, "toarray"):
                raw_expr = raw_expr.toarray()
            expr_dirs = np.array(raw_expr, dtype=np.float32)

        self.register_buffer("shape_dirs", torch.from_numpy(shape_dirs))
        self.register_buffer("expr_dirs",  torch.from_numpy(expr_dirs))

        # posedirs  (V*3, n_pose_blend)
        posedirs = np.array(flame_data["posedirs"], dtype=np.float32)
        self.register_buffer("posedirs", torch.from_numpy(posedirs))

        # faces  (F, 3) — integer indices
        faces = np.array(flame_data["f"], dtype=np.int64)
        self.register_buffer("faces", torch.from_numpy(faces))

        # J_regressor  (num_joints, V) — may be scipy sparse
        J_regressor = flame_data["J_regressor"]
        if hasattr(J_regressor, "toarray"):
            J_regressor = J_regressor.toarray()
        J_regressor = np.array(J_regressor, dtype=np.float32)
        self.register_buffer("J_regressor", torch.from_numpy(J_regressor))

        # LBS weights  (V, num_joints)
        weights = np.array(flame_data["weights"], dtype=np.float32)
        self.register_buffer("lbs_weights", torch.from_numpy(weights))

        # kintree_table  (2, num_joints)
        kintree = np.array(flame_data["kintree_table"], dtype=np.int64)
        self.register_buffer("kintree_table", torch.from_numpy(kintree))

    def _init_zeros(self) -> None:
        """Register zero-valued buffers so the module works without a real model file."""
        V = self.NUM_VERTS
        F = self.NUM_FACES
        num_joints = 5  # FLAME has 5 joints (head + jaw + 3 eye balls)

        self.register_buffer("v_template",   torch.zeros(V, 3))
        self.register_buffer("shape_dirs",   torch.zeros(V, 3, self.SHAPE_DIM))
        self.register_buffer("expr_dirs",    torch.zeros(V, 3, self.EXPR_DIM))
        self.register_buffer("posedirs",     torch.zeros(V * 3, 9 * (num_joints - 1)))
        self.register_buffer("faces",        torch.zeros(F, 3, dtype=torch.long))
        self.register_buffer("J_regressor",  torch.zeros(num_joints, V))
        self.register_buffer("lbs_weights",  torch.zeros(V, num_joints))
        self.register_buffer("kintree_table", torch.zeros(2, num_joints, dtype=torch.long))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        shape_params:      torch.Tensor,  # (B, 300)
        expression_params: torch.Tensor,  # (B, 50)
        pose_params:       torch.Tensor,  # (B, 6)
    ) -> FLAMEOutput:
        """
        Linear blend skinning:
        1. Shape blend: v = v_template + shape_dirs @ shape_params
        2. Expression blend: v += expr_dirs @ expr_params
        3. Project landmarks

        Note: pose_params is accepted for API compatibility but not applied in this
        implementation. Shape-only optimization is the primary use case.
        Pass torch.zeros(B, 6) for pose.
        """
        assert shape_params.shape[-1] == self.SHAPE_DIM, \
            f"Expected shape_params dim {self.SHAPE_DIM}, got {shape_params.shape[-1]}"
        assert expression_params.shape[-1] == self.EXPR_DIM, \
            f"Expected expression_params dim {self.EXPR_DIM}, got {expression_params.shape[-1]}"

        B = shape_params.shape[0]
        V = self.v_template.shape[0]

        # Reshape blend shape matrices to (V*3, n_basis)
        sd = self.shape_dirs.reshape(V * 3, -1)   # (V*3, 300)
        ed = self.expr_dirs.reshape(V * 3, -1)    # (V*3, 50)

        # 1. Shape blend  (shape_params @ sd.T -> (B, V*3))
        v_shaped = self.v_template.unsqueeze(0) + (
            (shape_params @ sd.T).reshape(B, V, 3)
        )

        # 2. Expression blend
        v_shaped = v_shaped + (
            (expression_params @ ed.T).reshape(B, V, 3)
        )

        vertices = v_shaped  # (B, V, 3)

        # 3. Extract 3D landmarks
        lmk3d = vertices[:, self.landmark_indices]  # (B, 68, 3)

        # 4. Orthographic 2D projection normalised to [-1, 1]
        # FLAME coordinates are ~0.1 m scale
        lmk2d = lmk3d[..., :2] / 0.1  # (B, 68, 2)

        return FLAMEOutput(
            vertices=vertices,
            faces=self.faces,
            landmarks2d=lmk2d,
            landmarks3d=lmk3d,
        )

    # ------------------------------------------------------------------
    # Contour vertices
    # ------------------------------------------------------------------

    def get_contour_vertices(
        self,
        vertices: torch.Tensor,  # (B, 5023, 3)
    ) -> torch.Tensor:
        """
        Returns edge (jaw contour) vertices for contour-aware loss.
        Returns: (B, N_contour, 3)
        """
        return vertices[:, self.contour_indices]  # (B, 17, 3)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def faces_tensor(self) -> torch.Tensor:
        """(9976, 3) int64"""
        return self.faces
