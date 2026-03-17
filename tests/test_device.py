"""Tests for faceforge.utils.device"""

import torch
import pytest
from unittest.mock import patch


def test_get_device_returns_torch_device():
    """get_device() must return a torch.device instance."""
    from faceforge.utils.device import get_device

    device = get_device()
    assert isinstance(device, torch.device)


def test_env_override_cpu(monkeypatch):
    """FACEFORGE_DEVICE=cpu must force CPU regardless of available hardware."""
    from faceforge.utils.device import get_device

    monkeypatch.setenv("FACEFORGE_DEVICE", "cpu")
    device = get_device()
    assert device == torch.device("cpu")


def test_env_override_mps_if_available(monkeypatch):
    """FACEFORGE_DEVICE=mps must be honoured when set, even if MPS is not
    actually available (we trust the env var — torch will raise later if not
    supported, not here)."""
    from faceforge.utils.device import get_device

    monkeypatch.setenv("FACEFORGE_DEVICE", "mps")
    device = get_device()
    assert device == torch.device("mps")


def test_env_override_cuda(monkeypatch):
    """FACEFORGE_DEVICE=cuda must be honoured when set."""
    from faceforge.utils.device import get_device

    monkeypatch.setenv("FACEFORGE_DEVICE", "cuda")
    device = get_device()
    assert device == torch.device("cuda")


def test_auto_detection_fallback_to_cpu(monkeypatch):
    """With no env var and no GPU, auto-detection must fall back to CPU."""
    from faceforge.utils.device import get_device

    monkeypatch.delenv("FACEFORGE_DEVICE", raising=False)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device = get_device()
    assert device == torch.device("cpu")


def test_to_device_moves_tensor():
    """to_device() must return a tensor on the requested device."""
    from faceforge.utils.device import to_device

    t = torch.zeros(3)
    result = to_device(t, torch.device("cpu"))
    assert result.device == torch.device("cpu")
    assert isinstance(result, torch.Tensor)
