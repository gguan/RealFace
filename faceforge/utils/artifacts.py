"""Intermediate artifact writer.

Saves stage-aligned intermediate outputs under a stable directory hierarchy:

  <output_dir>/intermediates/<subject_id>/
    01_input/          original input images
    02_preprocess/     canonical aligned crops + landmark previews
    03_mica/           MICA initial mesh + preview render
    04_refine/         refined mesh + preview render
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class ArtifactWriter:
    """Write intermediate pipeline artifacts to disk.

    When ``enabled=False`` all methods are no-ops so the pipeline code
    does not need to check the flag at every call site.
    """

    STAGES = ["01_input", "02_preprocess", "03_mica", "04_refine"]

    def __init__(self, output_dir: str, subject_id: str, enabled: bool = False):
        self.enabled = enabled
        self._base = Path(output_dir) / "intermediates" / subject_id

    def ensure_stage_dirs(self) -> None:
        """Create all stage directories (no-op if disabled)."""
        if not self.enabled:
            return
        for stage in self.STAGES:
            (self._base / stage).mkdir(parents=True, exist_ok=True)

    def _stage_path(self, stage: str) -> Path:
        return self._base / stage

    # ------------------------------------------------------------------
    # Stage 01: input
    # ------------------------------------------------------------------

    def save_input(self, image: np.ndarray, name: str = "input.png") -> None:
        """Save original input image."""
        if not self.enabled:
            return
        try:
            import cv2
            path = self._stage_path("01_input") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            logger.debug(f"[ArtifactWriter] Saved input: {path}")
        except Exception as exc:
            logger.warning(f"[ArtifactWriter] Failed to save input: {exc}")

    # ------------------------------------------------------------------
    # Stage 02: preprocessing
    # ------------------------------------------------------------------

    def save_aligned(self, aligned: np.ndarray, name: str = "aligned.png") -> None:
        """Save canonical aligned crop."""
        if not self.enabled:
            return
        try:
            import cv2
            path = self._stage_path("02_preprocess") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            logger.warning(f"[ArtifactWriter] Failed to save aligned: {exc}")

    def save_landmark_preview(
        self, preview: np.ndarray, name: str = "landmarks.png"
    ) -> None:
        """Save landmark overlay preview."""
        if not self.enabled:
            return
        try:
            import cv2
            path = self._stage_path("02_preprocess") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            logger.warning(f"[ArtifactWriter] Failed to save landmark preview: {exc}")

    # ------------------------------------------------------------------
    # Stage 03: MICA initial mesh
    # ------------------------------------------------------------------

    def save_mica_mesh(
        self,
        vertices: "np.ndarray",
        faces: "np.ndarray",
        name: str = "mica_initial.ply",
    ) -> None:
        """Save MICA initial mesh."""
        if not self.enabled:
            return
        try:
            from faceforge.utils.mesh_io import save_mesh
            path = str(self._stage_path("03_mica") / name)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            save_mesh(path, vertices, faces)
        except Exception as exc:
            logger.warning(f"[ArtifactWriter] Failed to save MICA mesh: {exc}")

    def save_mica_preview(
        self, render: np.ndarray, name: str = "mica_preview.png"
    ) -> None:
        """Save MICA render preview."""
        if not self.enabled:
            return
        try:
            import cv2
            path = self._stage_path("03_mica") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            logger.warning(f"[ArtifactWriter] Failed to save MICA preview: {exc}")

    # ------------------------------------------------------------------
    # Stage 04: refined mesh
    # ------------------------------------------------------------------

    def save_refined_mesh(
        self,
        vertices: "np.ndarray",
        faces: "np.ndarray",
        name: str = "refined.ply",
    ) -> None:
        """Save refined mesh."""
        if not self.enabled:
            return
        try:
            from faceforge.utils.mesh_io import save_mesh
            path = str(self._stage_path("04_refine") / name)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            save_mesh(path, vertices, faces)
        except Exception as exc:
            logger.warning(f"[ArtifactWriter] Failed to save refined mesh: {exc}")

    def save_refined_preview(
        self, render: np.ndarray, name: str = "refined_preview.png"
    ) -> None:
        """Save refined render preview."""
        if not self.enabled:
            return
        try:
            import cv2
            path = self._stage_path("04_refine") / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            logger.warning(f"[ArtifactWriter] Failed to save refined preview: {exc}")
