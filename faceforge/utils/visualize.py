"""Visualization utilities: mesh overlay, landmarks, loss curve."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger


def draw_landmarks(
    image: np.ndarray,  # (H, W, 3) uint8
    landmarks: np.ndarray,  # (68, 2) pixel or normalized coords
    color: Tuple[int, int, int] = (0, 255, 0),
    normalize: bool = False,
) -> np.ndarray:
    """Draw 68-point facial landmarks on image."""
    vis = image.copy()
    H, W = vis.shape[:2]
    lmks = landmarks.copy()
    if normalize:
        lmks[:, 0] *= W
        lmks[:, 1] *= H
    for x, y in lmks.astype(int):
        cv2.circle(vis, (x, y), 2, color, -1)
    return vis


def draw_mesh_overlay(
    image: np.ndarray,       # (H, W, 3) uint8
    render: np.ndarray,      # (H, W, 4) RGBA float [0,1]
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend rendered mesh RGBA onto original image."""
    vis = image.copy().astype(np.float32) / 255.0
    mask = render[..., 3:4]   # (H, W, 1)
    rgb  = render[..., :3]    # (H, W, 3)
    blended = vis * (1 - mask * alpha) + rgb * mask * alpha
    return (blended * 255).clip(0, 255).astype(np.uint8)


def save_loss_curve(
    loss_history: List[float],
    output_path: str,
) -> None:
    """Save optimization loss curve as PNG."""
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_history, linewidth=2, color="#2196F3")
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Loss")
        ax.set_title("Refinement Loss Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        logger.info(f"[visualize] Loss curve saved: {output_path}")
    except ImportError:
        logger.warning("[visualize] matplotlib not installed — loss curve not saved")


def save_comparison(
    images: List[np.ndarray],
    labels: List[str],
    output_path: str,
) -> None:
    """Save side-by-side comparison of N images."""
    assert len(images) == len(labels)
    h = max(img.shape[0] for img in images)
    panels = []
    for img, label in zip(images, labels):
        panel = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        bar = np.zeros((30, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(np.vstack([bar, panel]))
    combined = np.hstack(panels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    logger.info(f"[visualize] Comparison saved: {output_path}")
