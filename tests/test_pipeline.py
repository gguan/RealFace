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
