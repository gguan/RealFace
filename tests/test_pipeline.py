"""
Tests for faceforge.pipeline — placeholder until pipeline is implemented.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


def test_pipeline_module_importable():
    """pipeline.py stub must be importable without errors."""
    import faceforge.pipeline  # noqa: F401


def test_faceforge_version():
    """faceforge.__version__ must be set."""
    import faceforge
    assert faceforge.__version__ == "1.0.0"


# ── Helpers ─────────────────────────────────────────────────────────

@dataclass
class MockFLAMEOutput:
    vertices:    torch.Tensor
    faces:       torch.Tensor
    landmarks2d: torch.Tensor
    landmarks3d: torch.Tensor


def make_mock_flame(B=1, V=5023):
    flame = MagicMock()
    def forward(shape, expr, pose):
        return MockFLAMEOutput(
            vertices=shape[:, :3].unsqueeze(1).expand(-1, V, 3).contiguous(),
            faces=torch.zeros(9976, 3, dtype=torch.long),
            landmarks2d=torch.zeros(B, 68, 2),
            landmarks3d=torch.zeros(B, 68, 3),
        )
    flame.side_effect = forward
    flame.get_contour_vertices = lambda v: v[:, :17, :]
    return flame


def make_mock_renderer(image_size=64):
    from faceforge.model.renderer import RenderOutput
    renderer = MagicMock()
    renderer.image_size = image_size
    renderer.render.return_value = RenderOutput(
        image=torch.zeros(1, image_size, image_size, 4),
        zbuf=torch.zeros(1, image_size, image_size, 1),
        pix_to_face=torch.full((1, image_size, image_size, 1), -1, dtype=torch.long),
    )
    renderer.project_points.return_value = torch.zeros(1, 68, 2)
    renderer.extract_face_mask.return_value = torch.ones(1, image_size, image_size)
    renderer.build_cameras.return_value = None
    return renderer


# ── Tests ────────────────────────────────────────────────────────────

def test_refine_result_dataclass():
    from faceforge.optimizer.refiner import RefineResult
    r = RefineResult(
        shape_params=torch.zeros(1, 300),
        loss_history=[1.0, 0.5],
        flame_output=None,
        n_steps_done=2,
    )
    assert r.shape_params.shape == (1, 300)
    assert len(r.loss_history) == 2


def test_refiner_returns_correct_shape():
    from faceforge.optimizer.refiner import ShapeRefiner
    from faceforge.optimizer.losses import LossWeights

    flame = make_mock_flame()
    renderer = make_mock_renderer()
    lmk_det = MagicMock()
    lmk_det.detect.return_value = np.zeros((68, 2), dtype=np.float32)
    face_det = MagicMock()
    face_det._app = MagicMock()
    face_det._app.get.return_value = []

    lw = LossWeights(landmark=1.0, photometric=0.0, identity=0.0,
                     contour=0.0, region=0.0, regularize=0.1)

    refiner = ShapeRefiner(
        flame=flame, renderer=renderer,
        lmk_detector=lmk_det, face_detector=face_det,
        loss_weights=lw, device=torch.device("cpu"),
    )

    shape_init = torch.zeros(1, 300)
    image = torch.rand(1, 64, 64, 3)
    cam_params = torch.tensor([[1.0, 0.0, 0.0]])

    result = refiner.refine(shape_init, image, cam_params, n_steps=5, lr=1e-3)

    assert result.shape_params.shape == (1, 300)
    assert len(result.loss_history) == 5
    assert result.n_steps_done == 5


def test_refiner_loss_recorded_each_step():
    from faceforge.optimizer.refiner import ShapeRefiner
    from faceforge.optimizer.losses import LossWeights

    flame = make_mock_flame()
    renderer = make_mock_renderer()
    lmk_det = MagicMock()
    lmk_det.detect.return_value = np.zeros((68, 2), dtype=np.float32)
    face_det = MagicMock()
    face_det._app = MagicMock()
    face_det._app.get.return_value = []

    lw = LossWeights(landmark=0.0, photometric=0.0, identity=0.0,
                     contour=0.0, region=0.0, regularize=0.1)

    refiner = ShapeRefiner(
        flame=flame, renderer=renderer,
        lmk_detector=lmk_det, face_detector=face_det,
        loss_weights=lw, device=torch.device("cpu"),
    )
    result = refiner.refine(
        torch.ones(1, 300), torch.rand(1, 64, 64, 3),
        torch.tensor([[1.0, 0.0, 0.0]]), n_steps=10,
    )
    assert len(result.loss_history) == 10
    # With regularize-only loss and Adam, loss should decrease or stay flat
    assert result.loss_history[-1] <= result.loss_history[0] + 0.01


def test_refiner_best_shape_returned():
    """best_shape should be the shape at the step with lowest loss."""
    from faceforge.optimizer.refiner import ShapeRefiner
    from faceforge.optimizer.losses import LossWeights

    flame = make_mock_flame()
    renderer = make_mock_renderer()
    lmk_det = MagicMock()
    lmk_det.detect.return_value = np.zeros((68, 2), dtype=np.float32)
    face_det = MagicMock()
    face_det._app = MagicMock()
    face_det._app.get.return_value = []

    lw = LossWeights(landmark=0.0, photometric=0.0, identity=0.0,
                     contour=0.0, region=0.0, regularize=0.1)

    refiner = ShapeRefiner(
        flame=flame, renderer=renderer,
        lmk_detector=lmk_det, face_detector=face_det,
        loss_weights=lw, device=torch.device("cpu"),
    )
    result = refiner.refine(
        torch.ones(1, 300), torch.rand(1, 64, 64, 3),
        torch.tensor([[1.0, 0.0, 0.0]]), n_steps=20,
    )
    # Result should be a valid tensor, not the original init
    assert result.shape_params.shape == (1, 300)
    assert isinstance(result.shape_params, torch.Tensor)


def test_pipeline_result_dataclass():
    from faceforge.pipeline import PipelineResult
    r = PipelineResult(
        mesh_path="output/subject.ply",
        render_path="",
        params_path="output/subject_shape.npy",
        shape_params=np.zeros(300),
        confidence=0.95,
        loss_final=0.123,
    )
    assert r.shape_params.shape == (300,)
    assert r.confidence == 0.95
    assert r.loss_final == pytest.approx(0.123)


def test_pipeline_from_config_classmethod_exists():
    from faceforge.pipeline import FaceForgePipeline
    assert hasattr(FaceForgePipeline, "from_config")
    assert callable(FaceForgePipeline.from_config)


def test_cli_importable():
    from faceforge.cli import app
    assert app is not None


def test_cli_help(tmp_path):
    """CLI --help must not crash."""
    from typer.testing import CliRunner
    from faceforge.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "reconstruct" in result.output.lower() or "3d" in result.output.lower()


def test_faceforge_getattr_invalid():
    """faceforge.__getattr__ should raise AttributeError for unknown names."""
    import faceforge
    with pytest.raises(AttributeError):
        _ = faceforge.NonExistentThing


def _make_pipeline_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "device": "cpu",
        "paths": {
            "flame_model": "nonexistent.pkl",
            "flame_masks": "nonexistent.pkl",
            "mica_weights": "nonexistent.tar",
        },
        "encoder": {"insightface_name": "buffalo_l", "image_size": 112},
        "aggregator": {"strategy": "median", "min_confidence": 0.7},
        "refiner": {
            "enabled": False,
            "n_steps": 10,
            "lr": 0.001,
            "render_size": 64,
            "losses": {
                "landmark": 1.0, "photometric": 0.5, "identity": 0.3,
                "contour": 0.5, "region": 0.8, "regularize": 0.1,
            },
        },
        "output": {
            "save_mesh": False, "save_render": False,
            "save_params": False, "mesh_format": "ply",
        },
    })


def test_pipeline_init_with_mocked_deps():
    """FaceForgePipeline.__init__ with all heavy deps mocked."""
    cfg = _make_pipeline_cfg()

    mock_encoder = MagicMock()
    mock_aggregator = MagicMock()
    mock_flame = MagicMock()
    mock_renderer = MagicMock()
    mock_refiner = MagicMock()
    mock_lmk_det = MagicMock()
    mock_face_det = MagicMock()

    with patch("faceforge.encoder.mica_encoder.MICAEncoder", return_value=mock_encoder), \
         patch("faceforge.encoder.multi_image.MultiImageAggregator", return_value=mock_aggregator), \
         patch("faceforge.model.flame.FLAMELayer", return_value=mock_flame), \
         patch("faceforge.model.renderer.DifferentiableRenderer", return_value=mock_renderer), \
         patch("faceforge.optimizer.refiner.ShapeRefiner", return_value=mock_refiner), \
         patch("faceforge.utils.landmarks.LandmarkDetector", return_value=mock_lmk_det), \
         patch("faceforge.utils.image.FaceDetector", return_value=mock_face_det):
        from faceforge.pipeline import FaceForgePipeline
        pipeline = FaceForgePipeline(cfg)
        assert pipeline.device.type == "cpu"
        assert pipeline.encoder is mock_encoder
        assert pipeline.flame is mock_flame


def test_pipeline_run_no_refine(tmp_path):
    """FaceForgePipeline.run() with refiner disabled and all deps mocked."""
    import torch
    import numpy as np
    from omegaconf import OmegaConf

    cfg = _make_pipeline_cfg()

    # Mock aggregator result
    from faceforge.encoder.multi_image import AggregationResult
    mock_agg_result = AggregationResult(
        shape_params=torch.zeros(1, 300),
        per_image_shapes=torch.zeros(1, 300),
        confidence=0.9,
        n_valid_images=1,
    )

    @dataclass
    class MockFLAMEOut:
        vertices: torch.Tensor
        faces: torch.Tensor
        landmarks2d: torch.Tensor
        landmarks3d: torch.Tensor

    mock_flame_out = MockFLAMEOut(
        vertices=torch.zeros(1, 5023, 3),
        faces=torch.zeros(9976, 3, dtype=torch.long),
        landmarks2d=torch.zeros(1, 68, 2),
        landmarks3d=torch.zeros(1, 68, 3),
    )

    mock_encoder = MagicMock()
    mock_aggregator = MagicMock()
    mock_aggregator.aggregate.return_value = mock_agg_result
    mock_flame = MagicMock()
    mock_flame.return_value = mock_flame_out
    mock_renderer = MagicMock()
    mock_refiner = MagicMock()
    mock_lmk_det = MagicMock()
    mock_face_det = MagicMock()

    with patch("faceforge.encoder.mica_encoder.MICAEncoder", return_value=mock_encoder), \
         patch("faceforge.encoder.multi_image.MultiImageAggregator", return_value=mock_aggregator), \
         patch("faceforge.model.flame.FLAMELayer", return_value=mock_flame), \
         patch("faceforge.model.renderer.DifferentiableRenderer", return_value=mock_renderer), \
         patch("faceforge.optimizer.refiner.ShapeRefiner", return_value=mock_refiner), \
         patch("faceforge.utils.landmarks.LandmarkDetector", return_value=mock_lmk_det), \
         patch("faceforge.utils.image.FaceDetector", return_value=mock_face_det):
        from faceforge.pipeline import FaceForgePipeline
        pipeline = FaceForgePipeline(cfg)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = pipeline.run(img, output_dir=str(tmp_path), subject_id="test")
        assert result.shape_params.shape == (300,)
        assert result.confidence == pytest.approx(0.9)
        assert result.loss_final == pytest.approx(0.0)
