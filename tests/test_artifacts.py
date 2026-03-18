"""Tests for faceforge.utils.artifacts — ArtifactWriter (Tasks 12-15)."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Task 13: ArtifactWriter unit tests ───────────────────────────────────────

def test_artifact_writer_importable():
    from faceforge.utils.artifacts import ArtifactWriter  # noqa: F401


def test_artifact_writer_disabled_by_default(tmp_path):
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject")
    assert aw.enabled is False


def test_artifact_writer_creates_stage_directories(tmp_path):
    """ensure_stage_dirs() must create all four stage dirs when enabled."""
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject", enabled=True)
    aw.ensure_stage_dirs()
    for stage in ["01_input", "02_preprocess", "03_mica", "04_refine"]:
        assert (tmp_path / "intermediates" / "subject" / stage).is_dir(), \
            f"Missing stage dir: {stage}"


def test_artifact_writer_stage_dirs_noop_when_disabled(tmp_path):
    """ensure_stage_dirs() must NOT create any directories when disabled."""
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject", enabled=False)
    aw.ensure_stage_dirs()
    assert not (tmp_path / "intermediates").exists()


def test_artifact_writer_save_input_noop_when_disabled(tmp_path):
    """save_input() must be a no-op when disabled."""
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject", enabled=False)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    aw.save_input(img)  # should not raise, should not write file
    assert not (tmp_path / "intermediates").exists()


def test_artifact_writer_save_input_writes_file(tmp_path):
    """save_input() writes a PNG when enabled."""
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject", enabled=True)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with patch("cv2.imwrite") as mock_write:
        mock_write.return_value = True
        aw.save_input(img, name="input.png")
        mock_write.assert_called_once()
        call_path = mock_write.call_args[0][0]
        assert "01_input" in call_path
        assert call_path.endswith("input.png")


def test_artifact_writer_best_effort_on_cv2_error(tmp_path):
    """save_input() swallows exceptions — no crash when cv2.imwrite fails."""
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject", enabled=True)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with patch("cv2.imwrite", side_effect=OSError("disk full")):
        aw.save_input(img)  # must not propagate


def test_artifact_writer_save_aligned_writes_to_preprocess(tmp_path):
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject", enabled=True)
    with patch("cv2.imwrite") as mock_write:
        mock_write.return_value = True
        aw.save_aligned(np.zeros((112, 112, 3), dtype=np.uint8))
        call_path = mock_write.call_args[0][0]
        assert "02_preprocess" in call_path


def test_artifact_writer_save_mica_mesh_uses_save_mesh(tmp_path):
    from faceforge.utils.artifacts import ArtifactWriter
    aw = ArtifactWriter(str(tmp_path), "subject", enabled=True)
    verts = np.zeros((5023, 3), dtype=np.float32)
    faces = np.zeros((9976, 3), dtype=np.int64)
    with patch("faceforge.utils.mesh_io.save_mesh") as mock_save:
        aw.save_mica_mesh(verts, faces)
        mock_save.assert_called_once()
        call_path = mock_save.call_args[0][0]
        assert "03_mica" in call_path


# ── Task 12: config default and CLI flag ─────────────────────────────────────

def test_default_config_has_save_intermediates():
    """config/default.yaml must have output.save_intermediates = false."""
    from omegaconf import OmegaConf
    import os
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "default.yaml"
    )
    cfg = OmegaConf.load(cfg_path)
    assert hasattr(cfg.output, "save_intermediates"), \
        "output.save_intermediates missing from default.yaml"
    assert cfg.output.save_intermediates is False


def test_cli_has_save_intermediates_flag():
    """CLI must expose --save-intermediates option."""
    from typer.testing import CliRunner
    from faceforge.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["reconstruct", "--help"])
    assert "save-intermediates" in result.output


# ── Task 14-15: pipeline integration tests ───────────────────────────────────

def _make_artifact_pipeline_cfg(save_intermediates: bool):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "device": "cpu",
        "paths": {"flame_model": "x", "flame_masks": "y", "mica_weights": "z"},
        "encoder": {"insightface_name": "buffalo_l", "image_size": 112},
        "aggregator": {"strategy": "median", "min_confidence": 0.7},
        "refiner": {
            "enabled": False, "n_steps": 1, "lr": 1e-3, "render_size": 64,
            "losses": {
                "landmark": 1.0, "photometric": 0.0, "identity": 0.0,
                "contour": 0.0, "region": 0.0, "regularize": 0.1,
            },
        },
        "output": {
            "save_mesh": False, "save_render": False, "save_params": False,
            "mesh_format": "ply", "save_intermediates": save_intermediates,
        },
    })


def _run_pipeline_with_mocks(cfg, tmp_path):
    """Run FaceForgePipeline.run() with all heavy deps mocked."""
    from dataclasses import dataclass
    import torch
    import numpy as np
    from unittest.mock import patch, MagicMock
    from faceforge.preprocess.stage import PreprocessResult
    from faceforge.encoder.mica_adapter import MICAResult
    from faceforge.encoder.multi_image import AggregationResult

    preprocess_result = PreprocessResult(
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
        landmarks_68=np.zeros((68, 2), dtype=np.float32),
        landmarks_5=np.zeros((5, 2), dtype=np.float32),
        preview_image=np.zeros((112, 112, 3), dtype=np.uint8),
        metadata={},
    )
    mica_result = MICAResult(
        shape_code=torch.zeros(1, 300),
        initial_vertices=torch.zeros(1, 5023, 3),
        initial_faces=torch.zeros(9976, 3, dtype=torch.long),
        metadata={},
    )
    agg_result = AggregationResult(
        shape_params=torch.zeros(1, 300),
        per_image_shapes=torch.zeros(1, 300),
        confidence=1.0,
        n_valid_images=1,
    )

    @dataclass
    class MockFLAMEOut:
        vertices: torch.Tensor
        faces: torch.Tensor
        landmarks2d: torch.Tensor
        landmarks3d: torch.Tensor

    flame_out = MockFLAMEOut(
        vertices=torch.zeros(1, 5023, 3),
        faces=torch.zeros(9976, 3, dtype=torch.long),
        landmarks2d=torch.zeros(1, 68, 2),
        landmarks3d=torch.zeros(1, 68, 3),
    )

    with patch("faceforge.preprocess.stage.CanonicalPreprocessor") as pre_cls, \
         patch("faceforge.encoder.mica_adapter.MICAAdapter") as mica_cls, \
         patch("faceforge.encoder.multi_image.MultiImageAggregator") as agg_cls, \
         patch("faceforge.encoder.mica_encoder.MICAEncoder"), \
         patch("faceforge.model.flame.FLAMELayer") as flame_cls, \
         patch("faceforge.model.renderer.DifferentiableRenderer"), \
         patch("faceforge.optimizer.refiner.ShapeRefiner"), \
         patch("faceforge.utils.landmarks.LandmarkDetector"), \
         patch("faceforge.utils.image.FaceDetector"):
        pre_cls.return_value.run.return_value = preprocess_result
        mica_cls.return_value.run.return_value = mica_result
        agg_cls.return_value.aggregate.return_value = agg_result
        flame_cls.return_value.return_value = flame_out

        from faceforge.pipeline import FaceForgePipeline
        pipeline = FaceForgePipeline(cfg)
        pipeline.run(
            np.zeros((64, 64, 3), dtype=np.uint8),
            output_dir=str(tmp_path),
            subject_id="subj",
        )


def test_pipeline_no_intermediates_dir_when_disabled(tmp_path):
    """No 'intermediates' directory should be created when save_intermediates=False."""
    cfg = _make_artifact_pipeline_cfg(save_intermediates=False)
    _run_pipeline_with_mocks(cfg, tmp_path)
    assert not (tmp_path / "intermediates").exists()


def test_pipeline_intermediates_stage_dirs_created_when_enabled(tmp_path):
    """All four stage directories must be created when save_intermediates=True."""
    cfg = _make_artifact_pipeline_cfg(save_intermediates=True)
    _run_pipeline_with_mocks(cfg, tmp_path)
    for stage in ["01_input", "02_preprocess", "03_mica", "04_refine"]:
        assert (tmp_path / "intermediates" / "subj" / stage).is_dir(), \
            f"Missing stage dir: {stage}"
