"""
MICA encoder: maps face images to FLAME identity shape coefficients.

Internal flow:
  image → InsightFace antelopev2 (ArcFace) → 512-dim embedding
  → MappingNetwork → FLAME shape coefficients (1, 300)
"""

import logging
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MappingNetwork(nn.Module):
    """Maps 512-dim ArcFace embedding to 300-dim FLAME shape coefficients.

    Architecture matches the MICA checkpoint regressor:
      network: Linear(512,300) x4  +  output: Linear(300,300)
    Keys: regressor.network.{0-3}.{weight,bias}, regressor.output.{weight,bias}
    """

    def __init__(self):
        super().__init__()
        self.regressor = _MICARegressor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class _MICARegressor(nn.Module):
    """Internal regressor matching MICA checkpoint layout."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 300),
            nn.Linear(300, 300),
            nn.Linear(300, 300),
            nn.Linear(300, 300),
        )
        self.output = nn.Linear(300, 300)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.network(x))


class MICAEncoder:
    """
    MICA encoder wrapper.
    Internal flow:
      image → InsightFace detect+align → ArcFace feature extraction
      → Mapping Network → FLAME shape coefficients (1, 300)
    """

    def __init__(
        self,
        mica_weights_path: str,          # data/pretrained/mica.tar
        flame_model_path: str,           # data/pretrained/FLAME2020/ (dir, not used here)
        device=None,
        insightface_name: str = "antelopev2",
    ):
        """
        Load:
        1. InsightFace antelopev2 (ArcFace feature extraction)
        2. MICA mapping network (ArcFace feat → shape coefficients)
        """
        from faceforge.utils.device import get_device

        self.device = device or get_device()
        self._image_size = 112
        self._insightface_name = insightface_name

        # Setup mapping network
        self._mapping = MappingNetwork().to(self.device)
        self._mapping.eval()

        # Load weights if available
        weights_path = Path(mica_weights_path)
        if weights_path.exists():
            self._load_weights(weights_path)
        else:
            logger.warning(
                f"[MICAEncoder] Weights not found at {mica_weights_path}. "
                "Using random initialization. Download from https://zielon.github.io/mica/"
            )

        # Setup InsightFace detector
        self._setup_detector()

        logger.info(f"[MICAEncoder] Initialized on {self.device}")

    def _load_weights(self, path: Path):
        """Load MICA checkpoint (PyTorch ZIP format, saved with torch.save)."""
        try:
            ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
            # MICA checkpoint has flameModel.regressor.* keys
            if "flameModel" in ckpt:
                self._mapping.regressor.load_state_dict(ckpt["flameModel"], strict=False)
                logger.info(f"[MICAEncoder] Loaded weights from {path.name} (flameModel)")
            elif "state_dict" in ckpt:
                self._mapping.load_state_dict(ckpt["state_dict"], strict=False)
                logger.info(f"[MICAEncoder] Loaded weights from {path.name} (state_dict)")
            else:
                self._mapping.load_state_dict(ckpt, strict=False)
                logger.info(f"[MICAEncoder] Loaded weights from {path.name}")
        except Exception as e:
            logger.warning(f"[MICAEncoder] Failed to load weights: {e}")

    def _setup_detector(self):
        """Initialize InsightFace model for ArcFace feature extraction."""
        import insightface

        app = insightface.app.FaceAnalysis(
            name=self._insightface_name,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(224, 224))
        self._detector = app

    @torch.no_grad()
    def encode(
        self,
        image: Union[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Single image encode.
        Input: image path or RGB numpy array
        Output: shape_params (1, 300) float32, on self.device

        If face detection fails: raise ValueError("No face detected in image")
        """
        from faceforge.utils.image import load_image

        # Load image
        img = load_image(image)  # (H, W, 3) RGB uint8

        # Detect face and get embedding
        faces = self._detector.get(img)
        if not faces:
            raise ValueError("No face detected in image")

        # Get best face (highest score)
        face = max(faces, key=lambda f: f.det_score)

        # Get ArcFace embedding (512-dim, already extracted by InsightFace)
        if face.embedding is not None:
            embedding = (
                torch.from_numpy(face.embedding).float().unsqueeze(0).to(self.device)
            )  # (1, 512)
        else:
            # Fallback: zero embedding if InsightFace doesn't return embedding
            logger.warning("[MICAEncoder] No ArcFace embedding returned, using zeros")
            embedding = torch.zeros(1, 512, device=self.device)

        # Map to shape params
        shape_params = self._mapping(embedding)  # (1, 300)
        return shape_params

    @torch.no_grad()
    def encode_batch(
        self,
        images: List[Union[str, np.ndarray]],
    ) -> torch.Tensor:
        """
        Batch encode.
        Output: (N, 300) — per-image encode() results concatenated
        """
        results = [self.encode(img) for img in images]
        return torch.cat(results, dim=0)
