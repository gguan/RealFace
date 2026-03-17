"""Tests for faceforge.model.renderer — DifferentiableRenderer."""

import torch
import pytest


def test_renderer_importable():
    from faceforge.model import renderer  # must import even without pytorch3d


def test_render_output_dataclass():
    from faceforge.model.renderer import RenderOutput
    out = RenderOutput(
        image=torch.zeros(1, 64, 64, 4),
        zbuf=torch.zeros(1, 64, 64, 1),
        pix_to_face=torch.full((1, 64, 64, 1), -1, dtype=torch.long),
    )
    assert out.image.shape == (1, 64, 64, 4)


def test_renderer_init_no_crash():
    from faceforge.model.renderer import DifferentiableRenderer
    r = DifferentiableRenderer(image_size=64, device=torch.device("cpu"))
    assert r.image_size == 64


def test_render_returns_correct_shape_without_p3d():
    """Without pytorch3d, render() must still return correct-shaped dummy output."""
    from faceforge.model.renderer import DifferentiableRenderer, HAS_PYTORCH3D
    r = DifferentiableRenderer(image_size=64, device=torch.device("cpu"))
    B, V, F = 1, 5023, 9976
    verts = torch.randn(B, V, 3) * 0.1
    faces = torch.randint(0, V, (F, 3))
    out = r.render(verts, faces, cameras=None)
    assert out.image.shape == (B, 64, 64, 4)
    assert out.zbuf.shape == (B, 64, 64, 1)
    assert out.pix_to_face.shape == (B, 64, 64, 1)


def test_extract_face_mask():
    from faceforge.model.renderer import DifferentiableRenderer, RenderOutput
    r = DifferentiableRenderer(image_size=64)
    pix_to_face = torch.full((1, 64, 64, 1), -1, dtype=torch.long)
    pix_to_face[0, 10:20, 10:20, 0] = 0  # some pixels have face
    ro = RenderOutput(
        image=torch.zeros(1, 64, 64, 4),
        zbuf=torch.zeros(1, 64, 64, 1),
        pix_to_face=pix_to_face,
    )
    mask = r.extract_face_mask(ro)
    assert mask.shape == (1, 64, 64)
    assert mask[0, 15, 15].item() == True
    assert mask[0, 0, 0].item() == False


def test_project_points_fallback():
    from faceforge.model.renderer import DifferentiableRenderer
    r = DifferentiableRenderer(image_size=64)
    pts = torch.rand(1, 68, 3)
    result = r.project_points(pts, cameras=None)
    assert result.shape == (1, 68, 2)
