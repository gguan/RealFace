"""
Multi-image aggregator: combines per-image MICA predictions into a single
robust shape estimate (median / mean / trimmed_mean strategies).
"""

import logging
from dataclasses import dataclass
from typing import List, Union, Optional, Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    shape_params: torch.Tensor      # (1, 300) aggregated shape coefficients
    per_image_shapes: torch.Tensor  # (N, 300) per-image predictions
    confidence: float               # prediction consistency score [0, 1]
    n_valid_images: int             # number of images where face was detected


class MultiImageAggregator:
    """
    Multi-image shape aggregator.
    Recommended: provide 3-5 images of the same person from different angles.
    """

    def __init__(
        self,
        encoder: "MICAEncoder",
        strategy: Literal["median", "mean", "trimmed_mean"] = "median",
        min_confidence: float = 0.7,
    ):
        self._encoder = encoder
        self._strategy = strategy
        self._min_confidence = min_confidence

    def aggregate(
        self,
        images: List[Union[str, np.ndarray]],
        weights: Optional[List[float]] = None,
    ) -> AggregationResult:
        """
        Steps:
        1. For each image call encoder.encode() — skip (with warning) if ValueError
        2. Aggregate according to strategy
        3. Compute confidence = mean pairwise cosine similarity across predictions
        4. If confidence < min_confidence: print warning
        """
        if not images:
            raise ValueError("At least one image is required")

        # Encode each image, skip failures with warning
        valid_shapes = []
        n_valid = 0
        for i, img in enumerate(images):
            try:
                shape = self._encoder.encode(img)  # (1, 300)
                valid_shapes.append(shape)
                n_valid += 1
            except ValueError as e:
                logger.warning(f"[MultiImageAggregator] Skipping image {i}: {e}")

        if not valid_shapes:
            raise ValueError("No faces detected in any of the provided images")

        # Stack to (N, 300)
        all_shapes = torch.cat(valid_shapes, dim=0)  # (N, 300)

        # Aggregate
        if self._strategy == "median":
            aggregated = self._median(all_shapes)
        elif self._strategy == "mean":
            if weights and len(weights) == n_valid:
                w = torch.tensor(weights, dtype=torch.float32, device=all_shapes.device)
                w = w / w.sum()
                aggregated = (all_shapes * w.unsqueeze(1)).sum(0, keepdim=True)
            else:
                aggregated = all_shapes.mean(0, keepdim=True)
        elif self._strategy == "trimmed_mean":
            aggregated = self._trimmed_mean(all_shapes)
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

        # Compute confidence
        confidence = self._compute_confidence(all_shapes)

        if confidence < self._min_confidence:
            logger.warning(
                f"[MultiImageAggregator] Low confidence: {confidence:.3f} < {self._min_confidence:.3f}. "
                "Consider removing outlier images or using better-quality inputs."
            )

        return AggregationResult(
            shape_params=aggregated,
            per_image_shapes=all_shapes,
            confidence=confidence,
            n_valid_images=n_valid,
        )

    def _median(self, shapes: torch.Tensor) -> torch.Tensor:
        """Elementwise median across images"""
        return torch.median(shapes, dim=0).values.unsqueeze(0)

    def _trimmed_mean(self, shapes: torch.Tensor, trim: float = 0.2) -> torch.Tensor:
        """Remove top/bottom trim fraction then average"""
        N = shapes.shape[0]
        if N <= 2:
            return shapes.mean(0, keepdim=True)
        n_trim = max(1, int(N * trim))
        # Sort per dimension, trim top/bottom
        sorted_shapes, _ = torch.sort(shapes, dim=0)
        trimmed = sorted_shapes[n_trim: N - n_trim]
        return trimmed.mean(0, keepdim=True)

    def _compute_confidence(self, shapes: torch.Tensor) -> float:
        """
        Mean pairwise cosine similarity across all predictions. Range [0, 1].
        Higher = more consistent predictions across images.
        Single image: returns 1.0
        """
        N = shapes.shape[0]
        if N == 1:
            return 1.0
        # Normalize rows to unit vectors
        normed = torch.nn.functional.normalize(shapes, dim=1)  # (N, 300)
        # Compute pairwise cosine similarity matrix
        sim_matrix = normed @ normed.T  # (N, N)
        # Average off-diagonal elements
        mask = ~torch.eye(N, dtype=torch.bool, device=shapes.device)
        mean_sim = sim_matrix[mask].mean().item()
        # Clamp to [0, 1]; cosine similarity is in [-1, 1], map via (sim + 1) / 2
        return max(0.0, min(1.0, (mean_sim + 1.0) / 2.0))
