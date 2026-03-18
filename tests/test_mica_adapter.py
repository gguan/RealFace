"""Tests for faceforge.encoder.mica_adapter — thin MICA adapter contract."""
import numpy as np
import torch
import pytest


def test_mica_adapter_importable():
    from faceforge.encoder.mica_adapter import MICAAdapter  # noqa: F401


def test_mica_result_fields():
    from faceforge.encoder.mica_adapter import MICAResult

    result = MICAResult(
        shape_code=torch.zeros(1, 300),
        initial_vertices=torch.zeros(1, 5023, 3),
        initial_faces=torch.zeros(9976, 3, dtype=torch.long),
        metadata={},
    )
    assert result.shape_code.shape == (1, 300)
    assert result.initial_vertices.shape == (1, 5023, 3)
    assert result.initial_faces.shape == (9976, 3)


def test_mica_adapter_consumes_preprocess_result():
    from faceforge.preprocess.stage import PreprocessResult
    from faceforge.encoder.mica_adapter import MICAAdapter

    preprocess = PreprocessResult(
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
        landmarks_68=np.zeros((68, 2), dtype=np.float32),
        landmarks_5=np.zeros((5, 2), dtype=np.float32),
        preview_image=np.zeros((112, 112, 3), dtype=np.uint8),
        metadata={},
    )

    adapter = MICAAdapter.__new__(MICAAdapter)
    adapter._run_reference_mica = lambda x: torch.zeros(1, 300)
    adapter._decode_initial_mesh = lambda x: (
        torch.zeros(1, 5023, 3),
        torch.zeros(9976, 3, dtype=torch.long),
    )

    result = adapter.run(preprocess)
    assert result.shape_code.shape == (1, 300)
    assert result.initial_vertices.shape == (1, 5023, 3)
    assert result.initial_faces.shape == (9976, 3)


def test_mica_adapter_run_does_not_depend_on_internal_detection():
    from faceforge.encoder.mica_adapter import MICAAdapter
    assert "detect" not in MICAAdapter.run.__code__.co_names


def test_mica_adapter_init_does_not_load_models():
    """MICAAdapter.__init__ must not trigger heavy model loads."""
    from faceforge.encoder.mica_adapter import MICAAdapter
    adapter = MICAAdapter()
    assert adapter._encoder is None
    assert adapter._flame is None


def test_mica_result_shape_code_dtype():
    from faceforge.encoder.mica_adapter import MICAResult
    r = MICAResult(
        shape_code=torch.zeros(1, 300, dtype=torch.float32),
        initial_vertices=torch.zeros(1, 5023, 3),
        initial_faces=torch.zeros(9976, 3, dtype=torch.long),
        metadata={"test": True},
    )
    assert r.shape_code.dtype == torch.float32
    assert r.metadata["test"] is True
