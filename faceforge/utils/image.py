"""Image loading, face detection, alignment, and MICA preprocessing utilities."""

from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image
import cv2


class FaceDetector:
    def __init__(self, model_name: str = "buffalo_l"):
        """Load InsightFace detector (auto-downloads on first run)."""
        import insightface
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(224, 224))

    def detect(self, image: np.ndarray) -> Optional[dict]:
        """
        Input: BGR or RGB numpy array (H, W, 3).
        Output: {
            'bbox': [x1, y1, x2, y2],
            'kps': (5, 2) 5-point keypoints,
            'score': float confidence
        }
        Returns None if no face detected.
        """
        faces = self.app.get(image)
        if not faces:
            return None
        # Return highest-confidence face
        face = max(faces, key=lambda f: f.det_score)
        return {
            "bbox": face.bbox.tolist(),
            "kps": face.kps,
            "score": float(face.det_score),
        }

    def align_crop(
        self,
        image: np.ndarray,
        size: int = 112,
    ) -> Tuple[np.ndarray, float]:
        """
        Face alignment crop, returns (aligned_face, confidence).
        aligned_face: (112, 112, 3) uint8 RGB
        """
        from insightface.utils import face_align

        faces = self.app.get(image)
        if not faces:
            raise ValueError("No face detected in image")
        face = max(faces, key=lambda f: f.det_score)
        aligned = face_align.norm_crop(image, landmark=face.kps, image_size=size)
        return aligned, float(face.det_score)


def load_image(path: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Unified image loading, output (H, W, 3) RGB uint8."""
    if isinstance(path, np.ndarray):
        return path
    if isinstance(path, Image.Image):
        return np.array(path.convert("RGB"))
    # str path
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_for_mica(
    image: np.ndarray,
    detector: Optional["FaceDetector"] = None,
) -> np.ndarray:
    """
    MICA input preprocessing: align crop → (112, 112, 3) → normalize to [-1, 1]
    Returns float32 numpy array. If detector is None, creates a temporary one (slow).
    """
    if detector is None:
        detector = FaceDetector()
    aligned, confidence = detector.align_crop(image, size=112)
    if aligned is None:
        # Fallback: just resize if no face detected
        aligned = cv2.resize(image, (112, 112))
    img = aligned.astype(np.float32)
    return (img / 127.5) - 1.0


# Alias for plan compatibility
preprocess_image = preprocess_for_mica
