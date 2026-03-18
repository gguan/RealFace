import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
from faceforge.optimizer.losses import (
    LossWeights,
    landmark_loss,
    photometric_loss,
    contour_loss,
    extract_face_contour,
    shape_regularizer,
    RegionWeightedLoss,
    REGION_WEIGHTS,
)

B, H, W = 2, 64, 64


def test_loss_weights_defaults():
    lw = LossWeights()
    assert lw.landmark == 1.0
    assert lw.regularize == 0.1


def test_landmark_loss_zero_when_equal():
    lmk = torch.rand(B, 68, 2)
    assert landmark_loss(lmk, lmk).item() == pytest.approx(0.0, abs=1e-6)


def test_landmark_loss_positive():
    pred = torch.zeros(B, 68, 2)
    gt   = torch.ones(B, 68, 2)
    assert landmark_loss(pred, gt).item() > 0


def test_landmark_loss_backward():
    pred = torch.zeros(B, 68, 2, requires_grad=True)
    gt   = torch.ones(B, 68, 2)
    loss = landmark_loss(pred, gt)
    loss.backward()
    assert pred.grad is not None


def test_photometric_loss_zero_when_equal():
    img  = torch.rand(B, H, W, 3)
    mask = torch.ones(B, H, W)
    assert photometric_loss(img, img, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_photometric_loss_masked():
    rendered = torch.ones(B, H, W, 3)
    target   = torch.zeros(B, H, W, 3)
    # Only top-left quarter is in mask
    mask = torch.zeros(B, H, W)
    mask[:, :H//2, :W//2] = 1.0
    loss = photometric_loss(rendered, target, mask)
    assert loss.item() > 0
    # Loss should be 1.0 since |1-0|=1 everywhere in mask
    assert abs(loss.item() - 1.0) < 1e-4


def test_photometric_loss_backward():
    rendered = torch.rand(B, H, W, 3, requires_grad=True)
    target   = torch.rand(B, H, W, 3)
    mask     = torch.ones(B, H, W)
    photometric_loss(rendered, target, mask).backward()
    assert rendered.grad is not None


def test_contour_loss_zero_when_equal():
    pts = torch.rand(B, 50, 2)
    loss = contour_loss(pts, pts)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_contour_loss_positive():
    p = torch.zeros(B, 10, 2)
    q = torch.ones(B, 10, 2)
    assert contour_loss(p, q).item() > 0


def test_contour_loss_backward():
    p = torch.rand(B, 20, 2, requires_grad=True)
    q = torch.rand(B, 30, 2)
    contour_loss(p, q).backward()
    assert p.grad is not None


def test_extract_face_contour_shape():
    mask = torch.zeros(B, H, W)
    mask[:, 10:50, 10:50] = 1.0  # square face region
    pts = extract_face_contour(mask, n_points=50)
    assert pts.shape == (B, 50, 2)


def test_extract_face_contour_range():
    mask = torch.zeros(1, H, W)
    mask[0, 10:50, 10:50] = 1.0
    pts = extract_face_contour(mask, n_points=20)
    # Coordinates should be in [-1, 1]
    assert pts.min().item() >= -1.0 - 1e-5
    assert pts.max().item() <= 1.0 + 1e-5


def test_shape_regularizer_zero_at_init():
    # Both terms are zero only when shape_params == shape_init == 0:
    # anchor_loss = mse(0, 0) = 0, prior_loss = 0.pow(2).mean() = 0
    s = torch.zeros(1, 300)
    assert shape_regularizer(s, s).item() == pytest.approx(0.0, abs=1e-6)


def test_shape_regularizer_positive():
    s1 = torch.zeros(1, 300)
    s2 = torch.ones(1, 300)
    assert shape_regularizer(s1, s2).item() > 0


def test_shape_regularizer_backward():
    s = torch.rand(1, 300, requires_grad=True)
    init = torch.zeros(1, 300)
    shape_regularizer(s, init).backward()
    assert s.grad is not None


def test_region_loss_fallback_no_masks():
    """Without FLAME masks file, falls back to uniform MSE."""
    loss_fn = RegionWeightedLoss("nonexistent/path.pkl")
    pred = torch.rand(1, 5023, 3)
    ref  = torch.rand(1, 5023, 3)
    loss = loss_fn(pred, ref)
    assert loss.item() >= 0
    # Compare to plain MSE
    expected = torch.nn.functional.mse_loss(pred, ref)
    assert abs(loss.item() - expected.item()) < 1e-5


def test_region_loss_backward():
    loss_fn = RegionWeightedLoss("nonexistent/path.pkl")
    pred = torch.rand(1, 5023, 3, requires_grad=True)
    ref  = torch.rand(1, 5023, 3)
    loss_fn(pred, ref).backward()
    assert pred.grad is not None


def test_identity_loss_preprocess_shape():
    """_preprocess must return (B, 3, 112, 112) in [-1, 1]."""
    from faceforge.optimizer.losses import IdentityLoss
    loss_fn = IdentityLoss(arcface_model=MagicMock())
    img = torch.rand(2, 64, 64, 3)  # BHWC [0,1]
    out = loss_fn._preprocess(img)
    assert out.shape == (2, 3, 112, 112)
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5


def test_identity_loss_forward_shape():
    """IdentityLoss.forward must return a scalar loss in [0, 2]."""
    from faceforge.optimizer.losses import IdentityLoss
    # Mock arcface returns unit embeddings
    mock_arcface = MagicMock()
    mock_arcface.return_value = torch.nn.functional.normalize(
        torch.rand(2, 512), dim=1
    )
    loss_fn = IdentityLoss(arcface_model=mock_arcface)
    rendered = torch.rand(2, 64, 64, 3)
    target   = torch.rand(2, 64, 64, 3)
    loss = loss_fn(rendered, target)
    assert loss.dim() == 0  # scalar
    assert -1e-5 <= loss.item() <= 2.0


def test_identity_loss_zero_when_identical():
    """Loss = 0 when rendered == target (embeddings are identical)."""
    from faceforge.optimizer.losses import IdentityLoss
    mock_arcface = MagicMock()
    emb = torch.nn.functional.normalize(torch.rand(1, 512), dim=1)
    mock_arcface.return_value = emb
    loss_fn = IdentityLoss(arcface_model=mock_arcface)
    img = torch.rand(1, 64, 64, 3)
    loss = loss_fn(img, img)
    # Same embedding → cosine_sim = 1 → loss = 0
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
