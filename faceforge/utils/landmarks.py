"""68-point 2-D facial landmark detector (face_alignment backend)."""

import numpy as np
import torch
from typing import Union


class LandmarkDetector:
    def __init__(self, device=None):
        """Load face_alignment detector."""
        import face_alignment

        # Use 'cpu' string even on MPS — face_alignment MPS support is limited
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device="cpu",
        )

    def detect(
        self,
        image: np.ndarray,  # (H, W, 3) RGB
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Returns 68 landmark coordinates.
        normalize=True: coordinates normalized to [0, 1]
        normalize=False: pixel coordinates
        shape: (68, 2)
        """
        landmarks = self._fa.get_landmarks_from_image(image)
        if landmarks is None or len(landmarks) == 0:
            raise ValueError("No landmarks detected in image")
        # Take first face, first two columns (x, y)
        pts = landmarks[0][:, :2].astype(np.float32)
        if normalize:
            pts = pts / np.array([image.shape[1], image.shape[0]], dtype=np.float32)
        return pts
