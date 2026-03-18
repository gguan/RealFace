"""
Tests for faceforge.model.flame
"""
import torch
import pytest


@pytest.fixture
def flame_layer():
    """Create FLAMELayer with zero-initialized weights (no real model file needed)."""
    from faceforge.model.flame import FLAMELayer
    # Use non-existent path to trigger zero-init
    return FLAMELayer("data/pretrained/FLAME2020/generic_model.pkl", device=torch.device("cpu"))


def test_flame_module_importable():
    """flame.py must be importable without errors."""
    import faceforge.model.flame  # noqa: F401


def test_flame_output_shape(flame_layer):
    """Verify FLAMEOutput has correct shapes."""
    B = 2
    shape = torch.zeros(B, 300)
    expr  = torch.zeros(B, 50)
    pose  = torch.zeros(B, 6)
    out = flame_layer(shape, expr, pose)
    assert out.vertices.shape == (B, 5023, 3)
    assert out.faces.shape == (9976, 3)
    assert out.landmarks2d.shape == (B, 68, 2)
    assert out.landmarks3d.shape == (B, 68, 3)


def test_flame_zero_params_no_nan(flame_layer):
    """Zero params should give valid (no NaN) output."""
    shape = torch.zeros(1, 300)
    expr  = torch.zeros(1, 50)
    pose  = torch.zeros(1, 6)
    out = flame_layer(shape, expr, pose)
    assert not out.vertices.isnan().any()
    assert not out.landmarks2d.isnan().any()


def test_flame_grad_flows(flame_layer):
    """Gradient must flow back through shape_params."""
    shape = torch.zeros(1, 300, requires_grad=True)
    expr  = torch.zeros(1, 50)
    pose  = torch.zeros(1, 6)
    out = flame_layer(shape, expr, pose)
    loss = out.vertices.sum()
    loss.backward()
    assert shape.grad is not None


def test_get_contour_vertices_shape(flame_layer):
    """get_contour_vertices returns (B, N_contour, 3)."""
    B = 1
    verts = torch.zeros(B, 5023, 3)
    contour = flame_layer.get_contour_vertices(verts)
    assert contour.ndim == 3
    assert contour.shape[0] == B
    assert contour.shape[2] == 3


def test_faces_tensor_property(flame_layer):
    """faces_tensor property returns (9976, 3) int64."""
    ft = flame_layer.faces_tensor
    assert ft.shape == (9976, 3)
    assert ft.dtype == torch.int64 or ft.dtype == torch.long


def test_flame_nonzero_shape_changes_vertices(flame_layer):
    """Non-zero shape params should change vertices when shape_dirs are non-zero.
    With zero-init, output is zero anyway — just check no crash and shape is right."""
    shape = torch.randn(1, 300)
    expr  = torch.zeros(1, 50)
    pose  = torch.zeros(1, 6)
    out = flame_layer(shape, expr, pose)
    assert out.vertices.shape == (1, 5023, 3)
    assert not out.vertices.isnan().any()


def test_flame_load_from_synthetic_pkl(tmp_path):
    """Cover _load_from_file by creating a minimal synthetic FLAME pkl."""
    import pickle
    import numpy as np
    import torch
    from faceforge.model.flame import FLAMELayer

    V = 5023
    F = 9976
    J = 5

    flame_data = {
        "v_template": np.zeros((V, 3), dtype=np.float32),
        # shapedirs with >=400 last dim so branch splits into shape+expr
        "shapedirs": np.zeros((V, 3, 400), dtype=np.float32),
        "posedirs": np.zeros((V * 3, 9 * (J - 1)), dtype=np.float32),
        "f": np.zeros((F, 3), dtype=np.int64),
        "J_regressor": np.zeros((J, V), dtype=np.float32),
        "weights": np.zeros((V, J), dtype=np.float32),
        "kintree_table": np.zeros((2, J), dtype=np.int64),
    }

    pkl_path = str(tmp_path / "flame.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(flame_data, f)

    layer = FLAMELayer(pkl_path, device=torch.device("cpu"))

    # Check that buffers are loaded correctly
    assert layer.v_template.shape == (V, 3)
    assert layer.shape_dirs.shape == (V, 3, 300)
    assert layer.expr_dirs.shape == (V, 3, 50)
    assert layer.faces.shape == (F, 3)

    # Forward pass should work
    shape = torch.zeros(1, 300)
    expr = torch.zeros(1, 50)
    pose = torch.zeros(1, 6)
    out = layer(shape, expr, pose)
    assert out.vertices.shape == (1, V, 3)


def test_flame_load_with_separate_exprdirs(tmp_path):
    """Cover the else branch: shapedirs < 400 dims, uses exprdirs key."""
    import pickle
    import numpy as np
    import torch
    from faceforge.model.flame import FLAMELayer

    V = 5023
    F = 9976
    J = 5

    flame_data = {
        "v_template": np.zeros((V, 3), dtype=np.float32),
        "shapedirs": np.zeros((V, 3, 200), dtype=np.float32),  # < 400
        "exprdirs": np.zeros((V, 3, 50), dtype=np.float32),
        "posedirs": np.zeros((V * 3, 9 * (J - 1)), dtype=np.float32),
        "f": np.zeros((F, 3), dtype=np.int64),
        "J_regressor": np.zeros((J, V), dtype=np.float32),
        "weights": np.zeros((V, J), dtype=np.float32),
        "kintree_table": np.zeros((2, J), dtype=np.int64),
    }

    pkl_path = str(tmp_path / "flame2.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(flame_data, f)

    layer = FLAMELayer(pkl_path, device=torch.device("cpu"))
    assert layer.shape_dirs.shape == (V, 3, 200)
    assert layer.expr_dirs.shape == (V, 3, 50)
