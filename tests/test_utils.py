"""Tests for faceforge.utils — mesh_io, image, landmarks."""
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─────────────────── mesh_io ────────────────────────────────────────────────

class TestMeshIO:
    def test_save_mesh_ply_roundtrip(self, tmp_path, simple_mesh):
        from faceforge.utils.mesh_io import save_mesh, load_mesh
        vertices, faces = simple_mesh
        path = str(tmp_path / "test.ply")
        saved = save_mesh(path, vertices, faces, format="ply")
        assert saved == path
        assert Path(path).exists()
        v2, f2 = load_mesh(path)
        assert v2.shape == vertices.shape
        assert f2.shape == faces.shape
        np.testing.assert_allclose(v2, vertices, atol=1e-5)

    def test_save_mesh_creates_parent_dirs(self, tmp_path, simple_mesh):
        from faceforge.utils.mesh_io import save_mesh
        vertices, faces = simple_mesh
        path = str(tmp_path / "nested" / "subdir" / "mesh.ply")
        save_mesh(path, vertices, faces)
        assert Path(path).exists()

    def test_save_mesh_returns_path(self, tmp_path, simple_mesh):
        from faceforge.utils.mesh_io import save_mesh
        vertices, faces = simple_mesh
        path = str(tmp_path / "m.ply")
        result = save_mesh(path, vertices, faces)
        assert result == path

    def test_load_mesh_returns_float32(self, tmp_path, simple_mesh):
        from faceforge.utils.mesh_io import save_mesh, load_mesh
        vertices, faces = simple_mesh
        path = str(tmp_path / "m.ply")
        save_mesh(path, vertices, faces)
        v, f = load_mesh(path)
        assert v.dtype == np.float32
        assert f.dtype == np.int32

    def test_save_mesh_obj_format(self, tmp_path, simple_mesh):
        from faceforge.utils.mesh_io import save_mesh, load_mesh
        vertices, faces = simple_mesh
        path = str(tmp_path / "test.obj")
        save_mesh(path, vertices, faces, format="obj")
        assert Path(path).exists()
        v2, f2 = load_mesh(path)
        assert v2.shape == vertices.shape


# ─────────────────── image ──────────────────────────────────────────────────

class TestLoadImage:
    def test_load_numpy_passthrough(self, random_face_image):
        from faceforge.utils.image import load_image
        result = load_image(random_face_image)
        assert result is random_face_image

    def test_load_pil_image(self):
        from PIL import Image
        from faceforge.utils.image import load_image
        pil_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
        result = load_image(pil_img)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_load_path_not_found(self, tmp_path):
        from faceforge.utils.image import load_image
        with pytest.raises(FileNotFoundError):
            load_image(str(tmp_path / "nonexistent.jpg"))

    def test_load_actual_file(self, tmp_path):
        """Save a small image to disk and load it via path."""
        import cv2
        from faceforge.utils.image import load_image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img_path = str(tmp_path / "face.jpg")
        cv2.imwrite(img_path, img)
        result = load_image(img_path)
        assert result.shape == (64, 64, 3)


# ─────────────────── landmarks ──────────────────────────────────────────────

def _make_landmark_detector_with_mock(mock_fa_inst):
    """Create a LandmarkDetector instance with face_alignment mocked in sys.modules."""
    import sys

    fa_mod = MagicMock()
    fa_mod.LandmarksType = MagicMock()
    fa_mod.LandmarksType.TWO_D = "2D"
    fa_mod.FaceAlignment.return_value = mock_fa_inst

    # We need to keep all existing modules in the patch to avoid re-import issues.
    # Also include faceforge.utils.landmarks to prevent it from being removed by patch cleanup.
    import faceforge.utils.landmarks as lm_mod
    extra = {"face_alignment": fa_mod, "faceforge.utils.landmarks": lm_mod}
    with patch.dict(sys.modules, extra):
        det = lm_mod.LandmarkDetector(device=None)
    # Keep the mock as _fa so detect() calls work outside the patch context
    det._fa = mock_fa_inst
    return det


class TestLandmarkDetector:
    """Tests with face_alignment mocked out to avoid heavy model download."""

    def test_detect_returns_68x2(self):
        mock_fa_inst = MagicMock()
        fake_pts = np.random.rand(68, 3).astype(np.float32)
        mock_fa_inst.get_landmarks_from_image.return_value = [fake_pts]

        det = _make_landmark_detector_with_mock(mock_fa_inst)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = det.detect(img, normalize=False)

        assert result.shape == (68, 2)
        assert result.dtype == np.float32

    def test_detect_normalize(self):
        mock_fa_inst = MagicMock()
        fake_pts = np.zeros((68, 3), dtype=np.float32)
        fake_pts[0] = [256.0, 128.0, 0.0]
        mock_fa_inst.get_landmarks_from_image.return_value = [fake_pts]

        det = _make_landmark_detector_with_mock(mock_fa_inst)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = det.detect(img, normalize=True)

        assert result.shape == (68, 2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(0.5)

    def test_detect_no_face_raises(self):
        mock_fa_inst = MagicMock()
        mock_fa_inst.get_landmarks_from_image.return_value = None

        det = _make_landmark_detector_with_mock(mock_fa_inst)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="No landmarks"):
            det.detect(img)

    def test_detect_empty_list_raises(self):
        mock_fa_inst = MagicMock()
        mock_fa_inst.get_landmarks_from_image.return_value = []

        det = _make_landmark_detector_with_mock(mock_fa_inst)
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            det.detect(img)
