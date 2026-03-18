"""Canonical preprocessing stage.

Produces one canonical 112×112 aligned face image, 68-point and 5-point landmarks,
and a preview image per input image — without any downstream-specific modifications.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger


@dataclass
class PreprocessResult:
    original_image: Optional[np.ndarray]   # (H, W, 3) uint8 original input
    aligned_image: Optional[np.ndarray]    # (112, 112, 3) uint8 canonical crop
    landmarks_68: Optional[np.ndarray]     # (68, 2) float32 pixel coords
    landmarks_5: Optional[np.ndarray]      # (5, 2)  float32 5-point kps from detector
    preview_image: Optional[np.ndarray]    # (112, 112, 3) uint8 landmarks drawn
    metadata: Dict[str, Any]


class CanonicalPreprocessor:
    """Detect face, produce one canonical 112×112 aligned crop + 68-pt landmarks.

    Lazy-loads InsightFace and face_alignment on first ``run()`` call.
    """

    def __init__(self, config=None, device=None):
        self._config = config
        self._device = device
        self._image_size = 112
        self._model_name = "antelopev2"
        if config is not None:
            self._image_size = getattr(config.encoder, "image_size", 112)
            self._model_name = getattr(config.encoder, "insightface_name", "antelopev2")
        self.face_detector = None
        self.landmark_detector = None

    def _ensure_initialized(self) -> None:
        if self.face_detector is not None:
            return
        from faceforge.utils.image import FaceDetector
        from faceforge.utils.landmarks import LandmarkDetector
        self.face_detector = FaceDetector(model_name=self._model_name)
        self.landmark_detector = LandmarkDetector(device=self._device)

    def _build_result(
        self,
        original_image: np.ndarray,
        aligned_image: np.ndarray,
        landmarks_68: np.ndarray,
        landmarks_5: np.ndarray,
        preview_image: np.ndarray,
        metadata: Dict[str, Any],
    ) -> PreprocessResult:
        return PreprocessResult(
            original_image=original_image,
            aligned_image=aligned_image,
            landmarks_68=landmarks_68,
            landmarks_5=landmarks_5,
            preview_image=preview_image,
            metadata=metadata,
        )

    def run(self, image: np.ndarray) -> PreprocessResult:
        """Preprocess a single RGB image.

        Returns a ``PreprocessResult`` with aligned 112×112 crop and landmarks.
        Falls back to zero arrays if face detection fails.
        """
        self._ensure_initialized()
        from faceforge.utils.image import detect_and_align_face

        aligned, info = detect_and_align_face(
            image, self.face_detector._app, self._image_size
        )

        if aligned is None:
            logger.warning("[CanonicalPreprocessor] No face detected; using zero fallback")
            aligned = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
            info = {}

        # 5-point landmarks from detector keypoints
        kps = info.get("kps", None) if info else None
        landmarks_5 = (
            np.array(kps, dtype=np.float32)
            if kps is not None
            else np.zeros((5, 2), dtype=np.float32)
        )

        # 68-point landmarks via face_alignment
        try:
            landmarks_68 = self.landmark_detector._detect_raw(aligned).astype(np.float32)
        except Exception as exc:
            logger.warning(f"[CanonicalPreprocessor] Landmark detection failed: {exc}")
            landmarks_68 = np.zeros((68, 2), dtype=np.float32)

        # Preview: draw landmarks
        try:
            from faceforge.utils.visualize import draw_landmarks
            preview = draw_landmarks(aligned, landmarks_68)
        except Exception:
            preview = aligned.copy()

        return self._build_result(
            original_image=image,
            aligned_image=aligned,
            landmarks_68=landmarks_68,
            landmarks_5=landmarks_5,
            preview_image=preview,
            metadata={
                "detection_score": float(info.get("score", 0.0)) if info else 0.0,
                "source": "canonical",
            },
        )
