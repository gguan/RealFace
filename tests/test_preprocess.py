"""Tests for faceforge.preprocess — canonical preprocessing stage contract."""
import numpy as np
import pytest


def test_preprocess_package_importable():
    import faceforge.preprocess  # noqa: F401


def test_preprocess_result_fields():
    from faceforge.preprocess.stage import PreprocessResult
    result = PreprocessResult(
        original_image=None,
        aligned_image=None,
        landmarks_68=None,
        landmarks_5=None,
        preview_image=None,
        metadata={},
    )
    assert hasattr(result, "aligned_image")
    assert hasattr(result, "landmarks_68")
    assert hasattr(result, "metadata")


def test_preprocessor_returns_canonical_aligned_image(monkeypatch):
    from faceforge.preprocess.stage import CanonicalPreprocessor

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    pre = CanonicalPreprocessor.__new__(CanonicalPreprocessor)
    pre.face_detector = None
    pre.landmark_detector = None

    result = pre._build_result(
        original_image=image,
        aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
        landmarks_68=np.zeros((68, 2), dtype=np.float32),
        landmarks_5=np.zeros((5, 2), dtype=np.float32),
        preview_image=np.zeros((112, 112, 3), dtype=np.uint8),
        metadata={"source": "test"},
    )

    assert result.aligned_image.shape == (112, 112, 3)
    assert result.landmarks_68.shape == (68, 2)
    assert result.landmarks_5.shape == (5, 2)


def test_preprocess_result_has_single_canonical_alignment():
    from faceforge.preprocess.stage import PreprocessResult
    fields = set(PreprocessResult.__dataclass_fields__.keys())
    assert "aligned_image" in fields
    assert "mica_aligned_image" not in fields
    assert "refine_aligned_image" not in fields


def test_canonical_preprocessor_init_does_not_load_models():
    """CanonicalPreprocessor.__init__ must not trigger heavy model loads."""
    from faceforge.preprocess.stage import CanonicalPreprocessor
    pre = CanonicalPreprocessor()
    assert pre.face_detector is None
    assert pre.landmark_detector is None


def test_canonical_preprocessor_build_result_roundtrip():
    """_build_result stores all fields faithfully."""
    from faceforge.preprocess.stage import CanonicalPreprocessor
    pre = CanonicalPreprocessor.__new__(CanonicalPreprocessor)
    orig = np.ones((480, 640, 3), dtype=np.uint8) * 42
    aligned = np.ones((112, 112, 3), dtype=np.uint8) * 7
    lmk68 = np.arange(68 * 2, dtype=np.float32).reshape(68, 2)
    lmk5 = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    preview = aligned.copy()

    result = pre._build_result(
        original_image=orig,
        aligned_image=aligned,
        landmarks_68=lmk68,
        landmarks_5=lmk5,
        preview_image=preview,
        metadata={"detection_score": 0.99},
    )

    assert result.original_image is orig
    assert result.aligned_image is aligned
    np.testing.assert_array_equal(result.landmarks_68, lmk68)
    np.testing.assert_array_equal(result.landmarks_5, lmk5)
    assert result.metadata["detection_score"] == pytest.approx(0.99)
