"""Thin MICA adapter operating on pre-aligned canonical images.

Takes a ``PreprocessResult`` (aligned 112×112 crop produced by
``CanonicalPreprocessor``) and returns FLAME shape codes.  Does **not**
run its own face detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from loguru import logger


@dataclass
class MICAResult:
    shape_code: torch.Tensor        # (1, 300) FLAME shape coefficients
    initial_vertices: torch.Tensor  # (1, 5023, 3)
    initial_faces: torch.Tensor     # (9976, 3) int64
    metadata: Dict[str, Any]


class MICAAdapter:
    """Thin MICA adapter: canonical aligned image → FLAME shape code.

    Wraps the ArcFace recognition model + mapping network.  Requires a
    ``PreprocessResult`` produced by ``CanonicalPreprocessor`` so that face
    detection is always a separate upstream concern.
    """

    def __init__(self, config=None, device=None):
        from faceforge.utils.device import get_device
        self._config = config
        self._device = device or get_device()
        self._encoder = None
        self._flame = None

    # ------------------------------------------------------------------
    # Lazy initialization helpers
    # ------------------------------------------------------------------

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        from faceforge.encoder.mica_encoder import MICAEncoder
        mica_path = self._config.paths.mica_weights if self._config else ""
        flame_path = self._config.paths.flame_model if self._config else ""
        name = (
            self._config.encoder.insightface_name
            if self._config else "antelopev2"
        )
        self._encoder = MICAEncoder(
            mica_path, flame_path, self._device, insightface_name=name
        )

    def _ensure_flame(self) -> None:
        if self._flame is not None:
            return
        if self._config is None:
            return
        from pathlib import Path
        flame_path = self._config.paths.flame_model
        if not Path(flame_path).exists():
            return
        try:
            from faceforge.model.flame import FLAMELayer
            self._flame = FLAMELayer(flame_path, self._device)
        except Exception as exc:
            logger.warning(f"[MICAAdapter] Could not load FLAME: {exc}")

    # ------------------------------------------------------------------
    # Internal computation steps (replaceable in tests via monkeypatching)
    # ------------------------------------------------------------------

    def _run_reference_mica(self, aligned_image: np.ndarray) -> torch.Tensor:
        """ArcFace + mapping network on a pre-aligned 112×112 crop.

        Returns ``(1, 300)`` shape code tensor.
        """
        self._ensure_encoder()
        enc = self._encoder

        # Get ArcFace embedding from the pre-aligned crop
        rec = enc._app.models.get("recognition")
        if rec is not None:
            raw_emb = rec.get_feat(aligned_image)
            emb = torch.from_numpy(raw_emb).unsqueeze(0).float().to(self._device)
        else:
            logger.warning("[MICAAdapter] No recognition model; using zero embedding")
            emb = torch.zeros(1, 512, device=self._device)

        with torch.no_grad():
            shape_code = enc._mapping(emb)  # (1, 300)
        return shape_code

    def _decode_initial_mesh(
        self, shape_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """FLAME forward at zero expression/pose.

        Returns ``(vertices (1, 5023, 3), faces (9976, 3) int64)``.
        """
        self._ensure_flame()
        if self._flame is None:
            B = shape_code.shape[0]
            verts = torch.zeros(B, 5023, 3, device=self._device)
            faces = torch.zeros(9976, 3, dtype=torch.long, device=self._device)
            return verts, faces

        exp = torch.zeros(1, 50, device=self._device)
        pose = torch.zeros(1, 6, device=self._device)
        with torch.no_grad():
            out = self._flame(shape_code, exp, pose)
        return out.vertices, out.faces

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, preprocess_result) -> MICAResult:
        """Encode a ``PreprocessResult`` into FLAME shape code + initial mesh.

        Requires ``preprocess_result.aligned_image``.  Does not call any
        internal face detection.
        """
        aligned = preprocess_result.aligned_image
        shape_code = self._run_reference_mica(aligned)
        vertices, faces = self._decode_initial_mesh(shape_code)
        return MICAResult(
            shape_code=shape_code,
            initial_vertices=vertices,
            initial_faces=faces,
            metadata={},
        )
