"""Device detection and tensor/module movement utilities."""

import os
import torch
from loguru import logger


def get_device() -> torch.device:
    """
    Priority: FACEFORGE_DEVICE env var > CUDA > MPS > CPU
    """
    env = os.environ.get("FACEFORGE_DEVICE", "").lower()
    if env in ("cuda", "mps", "cpu"):
        device = torch.device(env)
        logger.info(f"[device] forced via env: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"[device] auto-detected: {device}")
    return device


def to_device(obj, device: torch.device = None):
    """Move tensor or nn.Module to target device."""
    if device is None:
        device = get_device()
    return obj.to(device)
