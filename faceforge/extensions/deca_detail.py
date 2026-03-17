"""
DECA detail-layer extension.

Adds high-frequency geometric detail (wrinkles, pores) on top of the coarse
FLAME reconstruction produced by the main pipeline.

Interface (stub — not yet implemented):
"""

import numpy as np


class DECADetailLayer:
    """Wraps the DECA detail U-Net to produce a displacement map from a coarse mesh."""

    def __init__(self, checkpoint_path: str, device=None):
        """
        Load DECA detail-layer weights.

        Args:
            checkpoint_path: Path to the DECA detail checkpoint (.tar or .pth).
            device: torch.device to run on. If None, auto-detected.
        """
        raise NotImplementedError("DECADetailLayer is not yet implemented")

    def predict_detail(
        self,
        coarse_vertices: np.ndarray,   # (V, 3) coarse FLAME mesh
        image: np.ndarray,             # (H, W, 3) RGB input image
        shape_code: np.ndarray,        # (300,) FLAME shape coefficients
        exp_code: np.ndarray,          # (100,) FLAME expression coefficients
    ) -> np.ndarray:
        """
        Predict a per-vertex displacement map from the input image.

        Returns:
            detail_vertices: (V, 3) float32 — coarse_vertices + predicted displacement
        """
        raise NotImplementedError("DECADetailLayer.predict_detail is not yet implemented")

    def apply_to_mesh(
        self,
        coarse_vertices: np.ndarray,   # (V, 3)
        detail_vertices: np.ndarray,   # (V, 3)
        blend_weight: float = 1.0,
    ) -> np.ndarray:
        """
        Blend coarse and detail vertices.

        Args:
            blend_weight: 0.0 = coarse only, 1.0 = full detail.

        Returns:
            blended_vertices: (V, 3) float32
        """
        raise NotImplementedError("DECADetailLayer.apply_to_mesh is not yet implemented")
