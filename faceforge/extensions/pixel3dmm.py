"""
Pixel3DMM extension.

Uses Pixel3DMM's per-pixel normal and depth predictions to refine the
FLAME mesh surface after the main optimization loop.

Interface (stub — not yet implemented):
"""

import numpy as np
from typing import Optional, Tuple


class Pixel3DMMRefiner:
    """Wraps Pixel3DMM inference to provide dense normal/depth supervision."""

    def __init__(self, checkpoint_path: str, device=None):
        """
        Load Pixel3DMM model weights.

        Args:
            checkpoint_path: Path to the Pixel3DMM checkpoint directory.
            device: torch.device to run on. If None, auto-detected.
        """
        raise NotImplementedError("Pixel3DMMRefiner is not yet implemented")

    def predict_normals(
        self,
        image: np.ndarray,  # (H, W, 3) RGB uint8
    ) -> np.ndarray:
        """
        Predict per-pixel surface normals from a face image.

        Returns:
            normals: (H, W, 3) float32, values in [-1, 1]
        """
        raise NotImplementedError("Pixel3DMMRefiner.predict_normals is not yet implemented")

    def predict_depth(
        self,
        image: np.ndarray,  # (H, W, 3) RGB uint8
    ) -> np.ndarray:
        """
        Predict a dense depth map from a face image.

        Returns:
            depth: (H, W) float32, metric depth in metres (approximate)
        """
        raise NotImplementedError("Pixel3DMMRefiner.predict_depth is not yet implemented")

    def refine_mesh(
        self,
        vertices: np.ndarray,          # (V, 3) initial FLAME mesh
        faces: np.ndarray,             # (F, 3)
        image: np.ndarray,             # (H, W, 3) RGB
        camera_params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine mesh geometry using Pixel3DMM normal/depth predictions.

        Args:
            vertices: Initial mesh vertices.
            faces: Mesh faces (unchanged).
            image: Input RGB image.
            camera_params: Optional dict with keys 'focal', 'princpt', 'R', 't'.

        Returns:
            (refined_vertices, faces): Both as float32 / int32 numpy arrays.
        """
        raise NotImplementedError("Pixel3DMMRefiner.refine_mesh is not yet implemented")
