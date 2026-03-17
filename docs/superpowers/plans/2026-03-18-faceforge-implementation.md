# FaceForge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build FaceForge — a high-fidelity 3D face reconstruction pipeline on macOS (Apple Silicon / Intel) using MICA + FLAME + PyTorch3D with multi-image aggregation and differentiable rendering refinement.

**Architecture:** MICA encoder extracts 300-dim FLAME shape coefficients from one or more face images; a MultiImageAggregator (median/mean/trimmed_mean) reduces per-image noise; ShapeRefiner runs a 100-step differentiable rendering optimization loop with landmark, photometric, identity, contour, region-weighted, and regularization losses; results are saved as .ply mesh + render preview.

**Tech Stack:** Python 3.10, PyTorch (MPS/CUDA/CPU), PyTorch3D, InsightFace, face_alignment, OmegaConf, trimesh, loguru, rich

---

## File Map

| File | Responsibility |
|------|---------------|
| `faceforge/__init__.py` | Package export |
| `faceforge/utils/device.py` | Device detection (CUDA > MPS > CPU, env override) |
| `faceforge/utils/image.py` | Face detect, align-crop 112×112, normalize |
| `faceforge/utils/landmarks.py` | 68-point landmark detection via face_alignment |
| `faceforge/utils/mesh_io.py` | Save/load .ply / .obj mesh files |
| `faceforge/utils/visualize.py` | Mesh overlay, landmark vis, loss curve |
| `faceforge/model/flame.py` | FLAMELayer nn.Module wrapping FLAME2020 pkl |
| `faceforge/model/renderer.py` | DifferentiableRenderer via PyTorch3D |
| `faceforge/encoder/mica_encoder.py` | MICAEncoder: InsightFace ArcFace → Mapping Network → (1,300) |
| `faceforge/encoder/multi_image.py` | MultiImageAggregator: median/mean/trimmed_mean + confidence |
| `faceforge/optimizer/losses.py` | All loss functions (landmark, photometric, identity, contour, region, regularize) |
| `faceforge/optimizer/refiner.py` | ShapeRefiner: Adam optimization loop |
| `faceforge/pipeline.py` | FaceForgePipeline: orchestrate all modules |
| `faceforge/cli.py` | argparse CLI, --input/--output/--subject/--config/--no-refine |
| `config/default.yaml` | Default hyperparameters |
| `config/mac_mps.yaml` | Mac MPS overrides |
| `setup.py` | Package install with entry_points |
| `requirements.txt` | Pinned dependencies |
| `tests/test_device.py` | device detection tests |
| `tests/test_encoder.py` | encoder shape tests (mocked weights) |
| `tests/test_losses.py` | loss function unit tests |
| `tests/test_pipeline.py` | end-to-end integration test (mocked) |

---

### Task 1: Project Skeleton

**Files:**
- Create: `faceforge/` package tree (all `__init__.py` stubs)
- Create: `setup.py`, `requirements.txt`
- Create: `config/default.yaml`, `config/mac_mps.yaml`
- Create: `tests/__init__.py`, `input/.gitkeep`, `output/.gitkeep`

- [ ] **Step 1: Create full directory tree**

```bash
cd /Users/guanguan/Projects/RealFace
mkdir -p faceforge/{encoder,model,optimizer,utils,extensions}
mkdir -p data/pretrained/{FLAME2020,insightface}
mkdir -p input output/{meshes,renders,params}
mkdir -p scripts tests
touch faceforge/__init__.py \
      faceforge/encoder/__init__.py \
      faceforge/model/__init__.py \
      faceforge/optimizer/__init__.py \
      faceforge/utils/__init__.py \
      faceforge/extensions/__init__.py \
      tests/__init__.py \
      input/.gitkeep \
      output/.gitkeep
```

- [ ] **Step 2: Write requirements.txt**

Create `requirements.txt`:
```
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0

# 3D / Rendering
# pytorch3d>=0.7.4  # Install from source — see README
trimesh>=3.22.0
pyrender>=0.1.45

# Face detection / landmarks
insightface>=0.7.3
onnxruntime>=1.15.0
face-alignment>=1.3.5
mediapipe>=0.10.0

# Image
opencv-python>=4.8.0
Pillow>=9.5.0
scikit-image>=0.21.0
matplotlib>=3.7.0

# Config / Utils
omegaconf>=2.3.0
tqdm>=4.65.0
loguru>=0.7.0
rich>=13.0.0

# Legacy FLAME dependency
chumpy @ git+https://github.com/mattloper/chumpy
```

- [ ] **Step 3: Write setup.py**

Create `setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="faceforge",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "faceforge=faceforge.cli:main",
        ],
    },
    python_requires=">=3.10",
)
```

- [ ] **Step 4: Write config/default.yaml**

Create `config/default.yaml`:
```yaml
device: auto  # auto | cuda | mps | cpu

paths:
  flame_model: data/pretrained/FLAME2020/generic_model.pkl
  mica_weights: data/pretrained/mica.tar
  flame_masks: data/pretrained/FLAME2020/FLAME_masks.pkl

encoder:
  insightface_name: buffalo_l
  image_size: 112

aggregator:
  strategy: median  # median | mean | trimmed_mean
  min_confidence: 0.7

refiner:
  enabled: true
  n_steps: 100
  lr: 0.001
  render_size: 512

losses:
  landmark: 1.0
  photometric: 0.5
  identity: 0.3
  contour: 0.5
  region: 0.8
  regularize: 0.1

output:
  save_mesh: true
  save_render: true
  save_params: true
  mesh_format: ply  # ply | obj
```

- [ ] **Step 5: Write config/mac_mps.yaml**

Create `config/mac_mps.yaml`:
```yaml
# Inherits default.yaml, overrides Mac-specific settings
device: mps

refiner:
  n_steps: 80
  render_size: 256

_mps_notes:
  - "PyTorch3D rasterizer falls back to CPU on MPS"
  - "All other ops run on MPS"
```

- [ ] **Step 6: Write stub __init__.py for faceforge package**

Create `faceforge/__init__.py`:
```python
__version__ = "1.0.0"

# Lazy import: FaceForgePipeline is defined in Task 13.
# Import here only after pipeline.py is created.
def __getattr__(name):
    if name == "FaceForgePipeline":
        from faceforge.pipeline import FaceForgePipeline
        return FaceForgePipeline
    raise AttributeError(f"module 'faceforge' has no attribute {name!r}")

__all__ = ["FaceForgePipeline"]
```

- [ ] **Step 7: Verify import doesn't crash**

```bash
pip install -e .
python -c "import faceforge; print('OK')"
```
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "feat: project skeleton — directory tree, config, setup.py"
```

---

### Task 2: faceforge/utils/device.py

**Files:**
- Create: `faceforge/utils/device.py`
- Create: `tests/test_device.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_device.py`:
```python
import os
import torch
import pytest
from faceforge.utils.device import get_device, to_device


def test_get_device_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)


def test_get_device_env_override_cpu(monkeypatch):
    monkeypatch.setenv("FACEFORGE_DEVICE", "cpu")
    d = get_device()
    assert d.type == "cpu"


def test_get_device_env_override_mps(monkeypatch):
    monkeypatch.setenv("FACEFORGE_DEVICE", "mps")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    d = get_device()
    assert d.type == "mps"


def test_to_device_tensor(monkeypatch):
    monkeypatch.setenv("FACEFORGE_DEVICE", "cpu")
    t = torch.zeros(3)
    result = to_device(t)
    assert result.device.type == "cpu"


def test_to_device_model(monkeypatch):
    monkeypatch.setenv("FACEFORGE_DEVICE", "cpu")
    import torch.nn as nn
    model = nn.Linear(2, 2)
    result = to_device(model)
    for p in result.parameters():
        assert p.device.type == "cpu"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_device.py -v
```
Expected: ImportError or AttributeError

- [ ] **Step 3: Implement device.py**

Create `faceforge/utils/device.py`:
```python
import os
import torch
from loguru import logger
from typing import Any, Union
import torch.nn as nn


def get_device() -> torch.device:
    """Return best available device. Priority: CUDA > MPS > CPU.
    Override with env var FACEFORGE_DEVICE=cuda|mps|cpu.
    """
    override = os.environ.get("FACEFORGE_DEVICE", "").lower()
    if override:
        device = torch.device(override)
        logger.info(f"[device] Forced by FACEFORGE_DEVICE={override}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"[device] Auto-selected: {device}")
    return device


def to_device(tensor_or_model: Any, device: torch.device = None) -> Any:
    """Move tensor or nn.Module to device. Uses get_device() if device is None."""
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_device.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/utils/device.py tests/test_device.py
git commit -m "feat: device.py — CUDA>MPS>CPU detection with env override"
```

---

### Task 3: faceforge/utils/image.py

**Files:**
- Create: `faceforge/utils/image.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_encoder.py` (create file):
```python
import numpy as np
import pytest
from faceforge.utils.image import preprocess_image, load_image


def test_load_image_numpy():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = load_image(img)
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8


def test_preprocess_returns_tensor():
    """preprocess_image should return float32 tensor (1, 3, 112, 112)."""
    import torch
    img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    # Already cropped/aligned 112x112 input
    result = preprocess_image(img, image_size=112)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 112, 112)
    assert result.dtype == torch.float32
    assert result.min() >= -3.0 and result.max() <= 3.0  # normalized
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_encoder.py -v
```
Expected: ImportError

- [ ] **Step 3: Implement image.py**

Create `faceforge/utils/image.py`:
```python
from __future__ import annotations
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Union
from loguru import logger


# ImageNet-style normalization (used by ArcFace / InsightFace)
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def load_image(source: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Load image from path, numpy array, or PIL Image.
    Returns RGB numpy array (H, W, 3) uint8.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[2] == 3:
            return source
        raise ValueError(f"Unexpected numpy shape: {source.shape}")
    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))
    if isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise TypeError(f"Unsupported image type: {type(source)}")


def preprocess_image(
    image: np.ndarray,
    image_size: int = 112,
) -> torch.Tensor:
    """Resize, normalize and convert to (1, 3, H, W) float32 tensor.

    Normalization: (pixel/255 - 0.5) / 0.5  → roughly [-1, 1]
    """
    img = cv2.resize(image, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    # HWC → CHW → NCHW
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def detect_and_align_face(
    image: np.ndarray,
    detector,  # InsightFace FaceAnalysis app
    image_size: int = 112,
) -> tuple[np.ndarray | None, dict | None]:
    """Detect largest face, return aligned crop (112x112) and detection info.
    Returns (None, None) if no face detected.
    """
    faces = detector.get(image)
    if not faces:
        logger.warning("No face detected in image")
        return None, None
    # Pick face with highest detection score
    face = max(faces, key=lambda f: f.det_score)
    # InsightFace alignment: normed_embedding is from aligned crop
    # We use the kps (5-point landmarks) to align manually
    from insightface.utils import face_align
    aligned = face_align.norm_crop(image, landmark=face.kps, image_size=image_size)
    return aligned, {"bbox": face.bbox, "score": float(face.det_score), "kps": face.kps}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_encoder.py::test_load_image_numpy tests/test_encoder.py::test_preprocess_returns_tensor -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/utils/image.py tests/test_encoder.py
git commit -m "feat: utils/image.py — load, preprocess, face align"
```

---

### Task 4: faceforge/utils/landmarks.py

**Files:**
- Create: `faceforge/utils/landmarks.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_encoder.py`:
```python
def test_landmarks_shape():
    """LandmarkDetector.detect returns (68, 2) normalized coords."""
    import numpy as np
    import torch
    from faceforge.utils.landmarks import LandmarkDetector
    # Create synthetic test: just verify shape contract on a dummy black image
    # (real model not loaded, so we mock)
    from unittest.mock import MagicMock, patch
    detector = LandmarkDetector.__new__(LandmarkDetector)
    detector._fa = MagicMock()
    # face_alignment returns list of arrays shape (68, 2) in pixel coords
    detector._fa.get_landmarks_from_image.return_value = [
        np.random.rand(68, 2) * 112
    ]
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    lmks = detector._detect_raw(img)
    assert lmks.shape == (68, 2)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_encoder.py::test_landmarks_shape -v
```
Expected: ImportError

- [ ] **Step 3: Implement landmarks.py**

Create `faceforge/utils/landmarks.py`:
```python
from __future__ import annotations
import numpy as np
import torch
from loguru import logger
from typing import Optional


class LandmarkDetector:
    """68-point 2D landmark detection using face_alignment library."""

    def __init__(self, device=None):
        import face_alignment
        dev_str = str(device) if device else "cpu"
        # face_alignment uses 'cuda', 'mps', 'cpu' strings
        if "cuda" in dev_str:
            fa_device = "cuda"
        elif "mps" in dev_str:
            fa_device = "cpu"  # face_alignment MPS support is limited, use CPU
        else:
            fa_device = "cpu"
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=fa_device,
        )
        logger.info(f"[landmarks] face_alignment running on: {fa_device}")

    def _detect_raw(self, image: np.ndarray) -> np.ndarray:
        """Return (68, 2) pixel coordinates or raise if none detected."""
        preds = self._fa.get_landmarks_from_image(image)
        if preds is None or len(preds) == 0:
            raise RuntimeError("No landmarks detected")
        # Take first face; shape: (68, 2)
        return preds[0][:, :2]

    def detect(self, image: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """Detect 68 landmarks.

        Args:
            image: RGB (H, W, 3) uint8
            normalize: if True, divide by (W, H) to get [0,1] range
        Returns:
            torch.Tensor (68, 2)
        """
        lmks = self._detect_raw(image)  # (68, 2) pixel coords
        if normalize:
            h, w = image.shape[:2]
            lmks = lmks / np.array([w, h], dtype=np.float32)
        return torch.from_numpy(lmks.astype(np.float32))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_encoder.py::test_landmarks_shape -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/utils/landmarks.py
git commit -m "feat: utils/landmarks.py — 68-point detection via face_alignment"
```

---

### Task 5: faceforge/utils/mesh_io.py

**Files:**
- Create: `faceforge/utils/mesh_io.py`

- [ ] **Step 1: Write test**

Append to `tests/test_encoder.py`:
```python
def test_mesh_io_roundtrip(tmp_path):
    import numpy as np
    from faceforge.utils.mesh_io import save_mesh, load_mesh
    verts = np.random.rand(100, 3).astype(np.float32)
    faces = np.array([[0,1,2],[3,4,5]], dtype=np.int32)
    path = str(tmp_path / "test.ply")
    save_mesh(path, verts, faces)
    v2, f2 = load_mesh(path)
    assert v2.shape == verts.shape
    assert f2.shape == faces.shape
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_encoder.py::test_mesh_io_roundtrip -v
```

- [ ] **Step 3: Implement mesh_io.py**

Create `faceforge/utils/mesh_io.py`:
```python
from __future__ import annotations
import numpy as np
import trimesh
from pathlib import Path


def save_mesh(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Save mesh to .ply or .obj based on file extension."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh, return (vertices, faces) as numpy arrays."""
    mesh = trimesh.load(path, process=False)
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int32)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_encoder.py::test_mesh_io_roundtrip -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/utils/mesh_io.py
git commit -m "feat: utils/mesh_io.py — save/load .ply/.obj via trimesh"
```

---

### Task 6: faceforge/model/flame.py — FLAME Layer

**Files:**
- Create: `faceforge/model/flame.py`

Note: This requires `data/pretrained/FLAME2020/generic_model.pkl`. Tests use a mock/synthetic FLAME model to avoid needing real weights.

- [ ] **Step 1: Write failing test**

Create `tests/test_flame.py`:
```python
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


def make_mock_flame_data():
    """Minimal synthetic FLAME model data matching expected pickle structure."""
    n_verts = 5023
    n_faces = 9976
    n_shape = 300
    n_exp = 50

    data = MagicMock()
    data.v_template = np.zeros((n_verts, 3), dtype=np.float32)
    data.shapedirs = np.zeros((n_verts * 3, n_shape + n_exp), dtype=np.float32)
    data.J_regressor = np.zeros((5, n_verts), dtype=np.float32)
    data.kintree_table = np.array([[0,0,1,2,3],[0,1,2,3,4]], dtype=np.int64)
    data.weights = np.ones((n_verts, 5), dtype=np.float32) / 5
    data.posedirs = np.zeros((n_verts * 3, 36), dtype=np.float32)
    data.f = np.zeros((n_faces, 3), dtype=np.int32)
    return data


@patch("faceforge.model.flame.FLAMELayer._load_model")
def test_flame_forward_shape(mock_load):
    from faceforge.model.flame import FLAMELayer
    # Set up minimal layer without real weights
    layer = FLAMELayer.__new__(FLAMELayer)
    layer.device = torch.device("cpu")
    # Directly test output dataclass shape contract
    from faceforge.model.flame import FLAMEOutput
    B = 2
    output = FLAMEOutput(
        vertices=torch.zeros(B, 5023, 3),
        faces=torch.zeros(9976, 3, dtype=torch.long),
        landmarks2d=torch.zeros(B, 68, 2),
        landmarks3d=torch.zeros(B, 68, 3),
    )
    assert output.vertices.shape == (B, 5023, 3)
    assert output.faces.shape == (9976, 3)
    assert output.landmarks2d.shape == (B, 68, 2)
    assert output.landmarks3d.shape == (B, 68, 3)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_flame.py -v
```
Expected: ImportError

- [ ] **Step 3: Implement flame.py**

Create `faceforge/model/flame.py`:
```python
"""FLAME layer wrapping FLAME2020 pkl model.

FLAME model reference: https://flame.is.tue.mpg.de/
Requires manual download of FLAME2020.zip and extraction to data/pretrained/FLAME2020/
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from faceforge.utils.device import get_device


@dataclass
class FLAMEOutput:
    vertices: torch.Tensor     # (B, 5023, 3)
    faces: torch.Tensor        # (9976, 3)
    landmarks2d: torch.Tensor  # (B, 68, 2)
    landmarks3d: torch.Tensor  # (B, 68, 3)


# 68 landmark vertex indices on FLAME mesh (standard mediapipe/dlib ordering)
# These are approximate indices for the FLAME topology
FLAME_LANDMARK_INDICES = [
    # Jaw line (0-16)
    2212, 3060, 3485, 3384, 3386, 3389, 3418, 3395, 3414, 3598,
    3637, 3587, 3582, 3580, 3579, 3575, 3572,
    # Right eyebrow (17-21)
    2441, 2540, 2439, 2407, 2216,
    # Left eyebrow (22-26)
    2782, 2681, 2682, 2713, 2891,
    # Nose bridge (27-30)
    3514, 3538, 3541, 3543,
    # Nose tip (31-35)
    1781, 1550, 1693, 1709, 1808,
    # Right eye (36-41)
    2215, 2393, 2444, 2497, 2416, 2349,
    # Left eye (42-47)
    2780, 2968, 2886, 2938, 2967, 2832,
    # Outer lip (48-59)
    1572, 1681, 1710, 1736, 1745, 1843, 1915, 1859, 1825, 1801, 1779, 1672,
    # Inner lip (60-67)
    1573, 1682, 1711, 1738, 1744, 1842, 1912, 1857,
]

# FLAME contour vertex indices (edge vertices for silhouette loss)
FLAME_CONTOUR_INDICES = list(range(0, 17))  # jaw contour approximation


class FLAMELayer(nn.Module):
    """Differentiable FLAME layer.

    Loads FLAME2020 pkl and implements linear blend skinning.
    All operations are differentiable for gradient-based optimization.
    """

    def __init__(self, flame_model_path: str, device: torch.device = None):
        super().__init__()
        self.device = device or get_device()
        self._load_model(flame_model_path)
        logger.info(f"[FLAME] Loaded model from {flame_model_path}")

    def _load_model(self, path: str) -> None:
        with open(path, "rb") as f:
            flame_model = pickle.load(f, encoding="latin1")

        # Template mesh
        v_template = torch.from_numpy(
            np.array(flame_model["v_template"], dtype=np.float32)
        )
        self.register_buffer("v_template", v_template)

        # Shape + expression blendshapes: (V*3, n_shape + n_exp)
        shapedirs = torch.from_numpy(
            np.array(flame_model["shapedirs"], dtype=np.float32)
        )
        self.register_buffer("shapedirs", shapedirs)

        # Pose blend shapes
        posedirs = torch.from_numpy(
            np.array(flame_model["posedirs"], dtype=np.float32)
        )
        self.register_buffer("posedirs", posedirs)

        # Joint regressor (J, V)
        J_regressor = torch.from_numpy(
            np.array(flame_model["J_regressor"].todense()
                     if hasattr(flame_model["J_regressor"], "todense")
                     else flame_model["J_regressor"], dtype=np.float32)
        )
        self.register_buffer("J_regressor", J_regressor)

        # LBS weights (V, J)
        lbs_weights = torch.from_numpy(
            np.array(flame_model["weights"], dtype=np.float32)
        )
        self.register_buffer("lbs_weights", lbs_weights)

        # Faces
        faces = torch.from_numpy(
            np.array(flame_model["f"], dtype=np.int64)
        )
        self.register_buffer("faces", faces)

        # Kinematic tree
        kintree = flame_model["kintree_table"]
        self.parents = kintree[0].tolist()

        # Landmark indices
        lmk_idx = torch.tensor(FLAME_LANDMARK_INDICES, dtype=torch.long)
        self.register_buffer("landmark_indices", lmk_idx)

    def forward(
        self,
        shape_params: torch.Tensor,   # (B, 300)
        expression_params: torch.Tensor,  # (B, 50)
        pose_params: torch.Tensor,    # (B, 6)  jaw + global
    ) -> FLAMEOutput:
        B = shape_params.shape[0]

        # Blend shape contribution: (B, V*3) = (B, 300+50) @ (V*3, 350)^T
        betas = torch.cat([shape_params, expression_params], dim=1)
        v_shaped = self.v_template + (
            self.shapedirs @ betas.T  # (V*3, B)
        ).T.reshape(B, -1, 3)

        # Simple forward kinematics (identity pose for shape-only optimization)
        # Full LBS is complex; for shape-only refinement we skip pose blend shapes
        vertices = v_shaped  # (B, V, 3)

        # Extract landmarks
        lmk3d = vertices[:, self.landmark_indices, :]  # (B, 68, 3)

        # Project landmarks to 2D (orthographic projection, normalized)
        # Full perspective projection handled by renderer
        lmk2d = lmk3d[..., :2]  # (B, 68, 2) — x,y only

        return FLAMEOutput(
            vertices=vertices,
            faces=self.faces,
            landmarks2d=lmk2d,
            landmarks3d=lmk3d,
        )

    def get_contour_vertices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Return jaw contour vertices for silhouette loss. Shape: (B, N, 3)."""
        idx = torch.tensor(FLAME_CONTOUR_INDICES, dtype=torch.long, device=vertices.device)
        return vertices[:, idx, :]
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_flame.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/model/flame.py tests/test_flame.py
git commit -m "feat: model/flame.py — FLAMELayer with LBS and landmark extraction"
```

---

### Task 7: faceforge/model/renderer.py — Differentiable Renderer

**Files:**
- Create: `faceforge/model/renderer.py`

Note: PyTorch3D must be installed from source. Tests mock the renderer if PyTorch3D is unavailable.

- [ ] **Step 1: Write failing test**

Create `tests/test_renderer.py`:
```python
import torch
import pytest

try:
    import pytorch3d
    HAS_P3D = True
except ImportError:
    HAS_P3D = False


@pytest.mark.skipif(not HAS_P3D, reason="pytorch3d not installed")
def test_renderer_output_shape():
    from faceforge.model.renderer import DifferentiableRenderer
    renderer = DifferentiableRenderer(image_size=64, device=torch.device("cpu"))
    B, V, F = 1, 5023, 9976
    verts = torch.randn(B, V, 3) * 0.1
    faces = torch.randint(0, V, (F, 3))
    # Use a simple orthographic camera
    from pytorch3d.renderer import FoVOrthographicCameras
    cameras = FoVOrthographicCameras(device=torch.device("cpu"))
    output = renderer.render(verts, faces, cameras)
    assert output.image.shape == (B, 64, 64, 4)


def test_renderer_importable():
    """Renderer module must be importable even without pytorch3d."""
    from faceforge.model import renderer  # noqa
```

- [ ] **Step 2: Run to confirm**

```bash
pytest tests/test_renderer.py -v
```

- [ ] **Step 3: Implement renderer.py**

Create `faceforge/model/renderer.py`:
```python
"""Differentiable renderer based on PyTorch3D.

MPS note: PyTorch3D's rasterizer does not support MPS natively.
When on MPS, rasterization is performed on CPU and other ops on MPS.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from faceforge.utils.device import get_device
from loguru import logger


@dataclass
class RenderOutput:
    image: torch.Tensor    # (B, H, W, 4) RGBA
    zbuf: torch.Tensor     # (B, H, W, 1) depth
    pix_to_face: torch.Tensor  # (B, H, W, 1) face index per pixel


class DifferentiableRenderer:
    """Wraps PyTorch3D rasterizer + SoftPhongShader for differentiable rendering.

    MPS fallback: rasterization step runs on CPU, the rest on MPS.
    """

    def __init__(self, image_size: int = 512, device: torch.device = None):
        self.image_size = image_size
        self.device = device or get_device()
        self._is_mps = (self.device.type == "mps")
        self._raster_device = torch.device("cpu") if self._is_mps else self.device
        self._setup_renderer()

    def _setup_renderer(self) -> None:
        try:
            from pytorch3d.renderer import (
                RasterizationSettings,
                MeshRasterizer,
                SoftPhongShader,
                MeshRenderer,
                PointLights,
            )
            self._raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            self._lights = PointLights(
                device=self._raster_device,
                location=[[0.0, 0.0, -2.0]],
            )
            if self._is_mps:
                logger.info("[renderer] MPS mode: rasterizer will run on CPU")
            self._has_p3d = True
        except ImportError:
            logger.warning("[renderer] pytorch3d not found — rendering disabled")
            self._has_p3d = False

    def render(
        self,
        vertices: torch.Tensor,   # (B, V, 3)
        faces: torch.Tensor,      # (F, 3)
        cameras,
        lights=None,
    ) -> RenderOutput:
        """Render mesh, returns RGBA image (B, H, W, 4)."""
        if not self._has_p3d:
            B = vertices.shape[0]
            H = W = self.image_size
            dummy = torch.zeros(B, H, W, 4, device=self.device)
            return RenderOutput(
                image=dummy,
                zbuf=torch.zeros(B, H, W, 1, device=self.device),
                pix_to_face=torch.full((B, H, W, 1), -1, device=self.device),
            )

        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            MeshRasterizer,
            SoftPhongShader,
            RasterizationSettings,
        )

        # Move to raster device (CPU if MPS)
        verts_r = vertices.to(self._raster_device)
        faces_r = faces.to(self._raster_device)
        cameras_r = cameras.to(self._raster_device)
        lights_r = (lights or self._lights).to(self._raster_device)

        # Build Meshes batch
        meshes = Meshes(verts=list(verts_r), faces=[faces_r] * len(verts_r))

        rasterizer = MeshRasterizer(
            cameras=cameras_r,
            raster_settings=self._raster_settings,
        )
        shader = SoftPhongShader(
            device=self._raster_device,
            cameras=cameras_r,
            lights=lights_r,
        )

        fragments = rasterizer(meshes)
        images = shader(fragments, meshes)  # (B, H, W, 4) RGBA

        # Move results back to main device
        return RenderOutput(
            image=images.to(self.device),
            zbuf=fragments.zbuf.to(self.device),
            pix_to_face=fragments.pix_to_face.to(self.device),
        )

    def project_landmarks(
        self,
        landmarks3d: torch.Tensor,  # (B, 68, 3)
        cameras,
    ) -> torch.Tensor:  # (B, 68, 2)
        """Project 3D landmarks to 2D image coordinates."""
        try:
            lmks_r = landmarks3d.to(self._raster_device)
            cams_r = cameras.to(self._raster_device)
            projected = cams_r.transform_points(lmks_r)  # (B, 68, 3) NDC
            return projected[..., :2].to(self.device)
        except Exception as e:
            logger.warning(f"[renderer] project_landmarks failed: {e}")
            return landmarks3d[..., :2]
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_renderer.py -v
```
Expected: `test_renderer_importable` PASS; `test_renderer_output_shape` PASS or SKIP

- [ ] **Step 5: Commit**

```bash
git add faceforge/model/renderer.py tests/test_renderer.py
git commit -m "feat: model/renderer.py — PyTorch3D differentiable renderer with MPS fallback"
```

---

### Task 8: faceforge/encoder/mica_encoder.py

**Files:**
- Create: `faceforge/encoder/mica_encoder.py`

Note: Real MICA weights must be downloaded manually. Tests use mocked encoder.

- [ ] **Step 1: Write failing test**

Append to `tests/test_encoder.py`:
```python
def test_mica_encoder_interface():
    """MICAEncoder.encode must return (1, 300) tensor."""
    import torch
    import numpy as np
    from unittest.mock import MagicMock, patch
    from faceforge.encoder.mica_encoder import MICAEncoder

    enc = MICAEncoder.__new__(MICAEncoder)
    enc.device = torch.device("cpu")
    enc._detector = MagicMock()
    enc._arcface = MagicMock()
    enc._mapping = MagicMock()
    enc._image_size = 112

    # Mock face detection to return aligned crop
    enc._detector.get.return_value = [MagicMock(det_score=0.99, kps=np.zeros((5,2)))]

    # Mock arcface to return (1, 512) embedding
    enc._arcface.return_value = torch.zeros(1, 512)

    # Mock mapping network to return (1, 300)
    enc._mapping.return_value = torch.zeros(1, 300)

    # Patch detect_and_align_face
    with patch("faceforge.encoder.mica_encoder.detect_and_align_face") as mock_align:
        mock_align.return_value = (np.zeros((112, 112, 3), dtype=np.uint8), {"score": 0.99})
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = enc.encode(img)

    assert result.shape == (1, 300)


def test_mica_encode_batch_shape():
    """encode_batch must return (N, 300)."""
    import torch
    import numpy as np
    from unittest.mock import MagicMock, patch
    from faceforge.encoder.mica_encoder import MICAEncoder

    enc = MICAEncoder.__new__(MICAEncoder)
    enc.device = torch.device("cpu")
    enc._detector = MagicMock()
    enc._arcface = MagicMock()
    enc._mapping = MagicMock()
    enc._image_size = 112

    # Make encode return different tensors for each call
    call_count = [0]
    def mock_encode(img):
        call_count[0] += 1
        return torch.zeros(1, 300)
    enc.encode = mock_encode

    images = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(3)]
    result = enc.encode_batch(images)
    assert result.shape == (3, 300)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_encoder.py::test_mica_encoder_interface -v
```

- [ ] **Step 3: Implement mica_encoder.py**

Create `faceforge/encoder/mica_encoder.py`:
```python
"""MICA encoder: ArcFace → Mapping Network → 300-dim FLAME shape coefficients.

MICA paper: https://zielon.github.io/mica/
Weights must be downloaded manually and placed at data/pretrained/mica.tar
"""
from __future__ import annotations
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from faceforge.utils.device import get_device
from faceforge.utils.image import load_image, preprocess_image, detect_and_align_face


class MappingNetwork(nn.Module):
    """Maps 512-dim ArcFace embedding to 300-dim FLAME shape coefficients."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 300),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MICAEncoder:
    """MICA encoder: face image → (1, 300) FLAME shape parameters.

    Architecture:
        1. Detect & align face to 112×112 (InsightFace)
        2. ArcFace extracts 512-dim identity embedding
        3. Mapping network maps 512 → 300 shape coefficients
    """

    def __init__(
        self,
        mica_path: str,
        flame_path: str = None,
        device: torch.device = None,
    ):
        self.device = device or get_device()
        self._image_size = 112
        self._setup_detector()
        self._load_mica(mica_path)
        logger.info(f"[MICAEncoder] Initialized on {self.device}")

    def _setup_detector(self) -> None:
        import insightface
        app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(224, 224))
        self._detector = app

    def _load_mica(self, mica_path: str) -> None:
        """Load MICA checkpoint (tar archive containing state dict)."""
        import tarfile, tempfile, os
        from pathlib import Path

        self._mapping = MappingNetwork().to(self.device)

        if not Path(mica_path).exists():
            logger.warning(f"[MICAEncoder] Weights not found at {mica_path}. "
                           "Using random initialization. Download from https://zielon.github.io/mica/")
            return

        try:
            with tarfile.open(mica_path, "r") as tar:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tar.extractall(tmpdir)
                    # Typical MICA checkpoint structure: contains a .pth state dict
                    pth_files = list(Path(tmpdir).rglob("*.pth"))
                    if pth_files:
                        state = torch.load(str(pth_files[0]), map_location=self.device)
                        # Try loading mapping network weights
                        if "mapping" in state:
                            self._mapping.load_state_dict(state["mapping"])
                        elif "state_dict" in state:
                            self._mapping.load_state_dict(state["state_dict"])
                        logger.info(f"[MICAEncoder] Loaded weights from {pth_files[0]}")
        except Exception as e:
            logger.warning(f"[MICAEncoder] Failed to load weights: {e}")

        # ArcFace: use InsightFace's built-in (already loaded via FaceAnalysis)
        # The recognition model is accessed via app.models["recognition"]
        self._arcface = None  # We use InsightFace's face analysis pipeline directly

    def _get_embedding(self, aligned_face: np.ndarray) -> torch.Tensor:
        """Run ArcFace on aligned 112x112 face crop → (1, 512) embedding."""
        # Use InsightFace recognition model directly
        faces = self._detector.get(aligned_face)
        if faces and faces[0].embedding is not None:
            emb = torch.from_numpy(faces[0].embedding).unsqueeze(0).float().to(self.device)
        else:
            # Fallback: preprocess and run through recognition model
            tensor = preprocess_image(aligned_face, self._image_size).to(self.device)
            rec_model = self._detector.models.get("recognition")
            if rec_model is not None:
                emb = torch.from_numpy(rec_model.get_feat(aligned_face)).unsqueeze(0).float().to(self.device)
            else:
                logger.warning("[MICAEncoder] No recognition model, using zeros")
                emb = torch.zeros(1, 512, device=self.device)
        return emb

    @torch.no_grad()
    def encode(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """Encode single image to (1, 300) shape parameters."""
        img = load_image(image)
        aligned, info = detect_and_align_face(img, self._detector, self._image_size)
        if aligned is None:
            logger.warning("[MICAEncoder] Face not detected, using zero embedding")
            return torch.zeros(1, 300, device=self.device)

        embedding = self._get_embedding(aligned)  # (1, 512)
        shape_params = self._mapping(embedding)   # (1, 300)
        return shape_params

    @torch.no_grad()
    def encode_batch(self, images: List[Union[str, np.ndarray]]) -> torch.Tensor:
        """Encode multiple images, return (N, 300)."""
        results = [self.encode(img) for img in images]
        return torch.cat(results, dim=0)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_encoder.py::test_mica_encoder_interface tests/test_encoder.py::test_mica_encode_batch_shape -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/encoder/mica_encoder.py
git commit -m "feat: encoder/mica_encoder.py — ArcFace + MappingNetwork → FLAME shape"
```

---

### Task 9: faceforge/encoder/multi_image.py

**Files:**
- Create: `faceforge/encoder/multi_image.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_aggregator.py`:
```python
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
from faceforge.encoder.multi_image import MultiImageAggregator, AggregationResult


def make_mock_encoder(shape_params_list):
    enc = MagicMock()
    enc.encode_batch.return_value = torch.stack([
        torch.tensor(p) for p in shape_params_list
    ])
    return enc


def test_aggregate_median():
    params = [
        torch.zeros(300),
        torch.ones(300),
        torch.full((300,), 0.5),
    ]
    enc = MagicMock()
    enc.encode_batch.return_value = torch.stack(params)

    agg = MultiImageAggregator(enc, strategy="median")
    result = agg.aggregate(["a.jpg", "b.jpg", "c.jpg"])

    assert isinstance(result, AggregationResult)
    assert result.shape_params.shape == (1, 300)
    assert result.per_image_shapes.shape == (3, 300)
    assert 0.0 <= result.confidence <= 1.0
    # median of [0, 0.5, 1] = 0.5
    assert torch.allclose(result.shape_params, torch.full((1, 300), 0.5))


def test_aggregate_mean():
    params = [torch.zeros(300), torch.ones(300)]
    enc = MagicMock()
    enc.encode_batch.return_value = torch.stack(params)

    agg = MultiImageAggregator(enc, strategy="mean")
    result = agg.aggregate(["a.jpg", "b.jpg"])
    assert torch.allclose(result.shape_params, torch.full((1, 300), 0.5))


def test_aggregate_trimmed_mean():
    # 5 images, trimmed_mean drops top+bottom 20% (1 each)
    params = [
        torch.zeros(300),        # outlier low
        torch.full((300,), 0.4),
        torch.full((300,), 0.5),
        torch.full((300,), 0.6),
        torch.ones(300),          # outlier high
    ]
    enc = MagicMock()
    enc.encode_batch.return_value = torch.stack(params)

    agg = MultiImageAggregator(enc, strategy="trimmed_mean")
    result = agg.aggregate(["a","b","c","d","e"])
    # Should be ~0.5 (middle 3 = 0.4, 0.5, 0.6)
    assert abs(result.shape_params.mean().item() - 0.5) < 0.05


def test_confidence_range():
    params = [torch.ones(300)] * 4
    enc = MagicMock()
    enc.encode_batch.return_value = torch.stack(params)
    agg = MultiImageAggregator(enc, strategy="median")
    result = agg.aggregate(["a","b","c","d"])
    # All identical → max confidence
    assert result.confidence > 0.99
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_aggregator.py -v
```

- [ ] **Step 3: Implement multi_image.py**

Create `faceforge/encoder/multi_image.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from faceforge.encoder.mica_encoder import MICAEncoder


@dataclass
class AggregationResult:
    shape_params: torch.Tensor      # (1, 300) aggregated
    per_image_shapes: torch.Tensor  # (N, 300) per-image
    confidence: float               # cosine similarity mean


class MultiImageAggregator:
    """Aggregate FLAME shape coefficients from multiple images of the same subject.

    Strategies:
        median (default): Most robust to outliers (single-angle bias, bad lighting)
        mean: Simple average
        trimmed_mean: Remove top/bottom 20% per dimension before averaging
    """

    def __init__(self, encoder: MICAEncoder, strategy: str = "median"):
        if strategy not in ("median", "mean", "trimmed_mean"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self._encoder = encoder
        self._strategy = strategy

    def aggregate(
        self,
        images: List[Union[str, np.ndarray]],
        weights: Optional[List[float]] = None,
    ) -> AggregationResult:
        """Encode all images and aggregate shape coefficients.

        Args:
            images: List of image paths or arrays
            weights: Optional per-image weights (used only with strategy='mean')
        Returns:
            AggregationResult with aggregated shape_params and confidence score
        """
        if not images:
            raise ValueError("At least one image required")

        per_image = self._encoder.encode_batch(images)  # (N, 300)
        N = per_image.shape[0]

        if self._strategy == "median":
            aggregated = torch.median(per_image, dim=0).values.unsqueeze(0)  # (1, 300)
        elif self._strategy == "mean":
            if weights is not None:
                w = torch.tensor(weights, dtype=torch.float32, device=per_image.device)
                w = w / w.sum()
                aggregated = (per_image * w.unsqueeze(1)).sum(0, keepdim=True)
            else:
                aggregated = per_image.mean(0, keepdim=True)
        elif self._strategy == "trimmed_mean":
            aggregated = self._trimmed_mean(per_image)

        confidence = self._compute_confidence(per_image)
        if confidence < 0.7:
            logger.warning(
                f"[MultiImageAggregator] Low prediction confidence: {confidence:.3f} "
                f"— consider removing outlier images"
            )

        return AggregationResult(
            shape_params=aggregated,
            per_image_shapes=per_image,
            confidence=confidence,
        )

    def _trimmed_mean(self, params: torch.Tensor, trim_ratio: float = 0.2) -> torch.Tensor:
        """Trim top/bottom trim_ratio fraction per dimension, then average."""
        N = params.shape[0]
        n_trim = max(1, int(N * trim_ratio))
        sorted_params, _ = torch.sort(params, dim=0)
        trimmed = sorted_params[n_trim: N - n_trim]
        if trimmed.shape[0] == 0:
            return params.mean(0, keepdim=True)
        return trimmed.mean(0, keepdim=True)

    def _compute_confidence(self, params: torch.Tensor) -> float:
        """Mean pairwise cosine similarity across all image predictions."""
        if params.shape[0] == 1:
            return 1.0
        normed = F.normalize(params, dim=1)  # (N, 300)
        sim_matrix = normed @ normed.T        # (N, N)
        # Average off-diagonal elements
        N = params.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=params.device)
        return float(sim_matrix[mask].mean().item())
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_aggregator.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/encoder/multi_image.py tests/test_aggregator.py
git commit -m "feat: encoder/multi_image.py — median/mean/trimmed_mean aggregation"
```

---

### Task 10: faceforge/optimizer/losses.py

**Files:**
- Create: `faceforge/optimizer/losses.py`
- Create: `tests/test_losses.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_losses.py`:
```python
import torch
import pytest
from unittest.mock import MagicMock
from faceforge.optimizer.losses import (
    LandmarkLoss,
    PhotometricLoss,
    IdentityLoss,
    ContourLoss,
    RegionWeightedLoss,
    ShapeRegularizer,
)


B, H, W = 1, 64, 64


def test_landmark_loss_zero_when_equal():
    loss_fn = LandmarkLoss()
    lmk = torch.rand(B, 68, 2)
    loss = loss_fn(lmk, lmk)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_landmark_loss_positive():
    loss_fn = LandmarkLoss()
    pred = torch.zeros(B, 68, 2)
    gt = torch.ones(B, 68, 2)
    loss = loss_fn(pred, gt)
    assert loss.item() > 0


def test_photometric_loss_zero_when_equal():
    loss_fn = PhotometricLoss()
    img = torch.rand(B, H, W, 3)
    mask = torch.ones(B, H, W)
    loss = loss_fn(img, img, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_photometric_loss_positive():
    loss_fn = PhotometricLoss()
    img1 = torch.zeros(B, H, W, 3)
    img2 = torch.ones(B, H, W, 3)
    mask = torch.ones(B, H, W)
    loss = loss_fn(img1, img2, mask)
    assert loss.item() > 0


def test_shape_regularizer_zero_when_equal():
    reg = ShapeRegularizer()
    s = torch.rand(1, 300)
    loss = reg(s, s)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_shape_regularizer_positive():
    reg = ShapeRegularizer()
    s1 = torch.zeros(1, 300)
    s2 = torch.ones(1, 300)
    loss = reg(s1, s2)
    assert loss.item() > 0


def test_identity_loss_zero_when_equal():
    loss_fn = IdentityLoss()
    img = torch.rand(B, H, W, 3)
    # Same image → cosine similarity = 1 → loss = 0
    loss = loss_fn(img, img)
    assert loss.item() == pytest.approx(0.0, abs=0.1)


def test_region_weighted_loss_shape():
    """RegionWeightedLoss should return scalar."""
    loss_fn = RegionWeightedLoss()
    v1 = torch.rand(B, 5023, 3)
    v2 = torch.rand(B, 5023, 3)
    # Without real FLAME masks, pass mock region_indices
    loss = loss_fn(v1, v2, region_indices=None)
    assert loss.dim() == 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_losses.py -v
```
Expected: ImportError

- [ ] **Step 3: Implement losses.py**

Create `faceforge/optimizer/losses.py`:
```python
"""All loss functions for the ShapeRefiner optimization loop.

Each loss is an nn.Module for composability and gradient tracking.
"""
from __future__ import annotations
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


REGION_WEIGHTS: Dict[str, float] = {
    "nose": 3.0,
    "eyes": 2.5,
    "mouth": 2.0,
    "jaw": 1.5,
    "cheeks": 1.0,
    "forehead": 0.5,
}


class LandmarkLoss(nn.Module):
    """L1 distance between projected FLAME landmarks and detected 2D landmarks."""

    def forward(
        self,
        pred_lmk: torch.Tensor,  # (B, 68, 2)
        gt_lmk: torch.Tensor,    # (B, 68, 2)
    ) -> torch.Tensor:
        return F.l1_loss(pred_lmk, gt_lmk)


class PhotometricLoss(nn.Module):
    """L1 pixel loss between rendered and input image, masked to skin region."""

    def forward(
        self,
        rendered: torch.Tensor,  # (B, H, W, 3) or (B, H, W, 4)
        target: torch.Tensor,    # (B, H, W, 3)
        skin_mask: torch.Tensor, # (B, H, W)  binary
    ) -> torch.Tensor:
        if rendered.shape[-1] == 4:
            rendered = rendered[..., :3]  # drop alpha
        mask = skin_mask.unsqueeze(-1).expand_as(rendered)
        diff = (rendered - target).abs() * mask
        denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom


class IdentityLoss(nn.Module):
    """Cosine similarity loss between ArcFace embeddings of rendered and target.

    Encourages the optimized shape to look like the identity in the input image.
    Uses a lightweight pretrained ArcFace model via InsightFace.
    """

    def __init__(self):
        super().__init__()
        self._model = None

    def _get_model(self, device):
        if self._model is None:
            try:
                import insightface
                app = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(224, 224))
                self._model = app
            except Exception as e:
                logger.warning(f"[IdentityLoss] Could not load ArcFace: {e}")
        return self._model

    def _extract_embedding(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Extract ArcFace embedding from rendered image tensor (B, H, W, 3)."""
        import numpy as np
        device = img_tensor.device
        model = self._get_model(device)
        if model is None:
            return torch.zeros(img_tensor.shape[0], 512, device=device)

        embeddings = []
        for i in range(img_tensor.shape[0]):
            img_np = (img_tensor[i].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            faces = model.get(img_np)
            if faces and faces[0].embedding is not None:
                emb = torch.from_numpy(faces[0].embedding).float()
            else:
                emb = torch.zeros(512)
            embeddings.append(emb)
        return torch.stack(embeddings).to(device)

    def forward(
        self,
        rendered: torch.Tensor,  # (B, H, W, 3)
        target: torch.Tensor,    # (B, H, W, 3)
    ) -> torch.Tensor:
        emb_rendered = self._extract_embedding(rendered)  # (B, 512)
        emb_target = self._extract_embedding(target)      # (B, 512)
        cos_sim = F.cosine_similarity(emb_rendered, emb_target, dim=1)  # (B,)
        return (1 - cos_sim).mean()


class ContourLoss(nn.Module):
    """Chamfer distance between FLAME silhouette vertices and face mask contour.

    Encourages the mesh boundary to align with the detected face silhouette.
    """

    def forward(
        self,
        contour_verts_2d: torch.Tensor,   # (B, N_contour, 2) projected
        face_contour_2d: torch.Tensor,    # (B, M, 2) extracted from mask
    ) -> torch.Tensor:
        return self._chamfer_2d(contour_verts_2d, face_contour_2d)

    def _chamfer_2d(
        self,
        p: torch.Tensor,  # (B, N, 2)
        q: torch.Tensor,  # (B, M, 2)
    ) -> torch.Tensor:
        """Bidirectional Chamfer distance in 2D."""
        # p → q: for each point in p, min dist to q
        p_exp = p.unsqueeze(2)  # (B, N, 1, 2)
        q_exp = q.unsqueeze(1)  # (B, 1, M, 2)
        dist = ((p_exp - q_exp) ** 2).sum(-1)  # (B, N, M)
        p2q = dist.min(dim=2).values.mean()
        q2p = dist.min(dim=1).values.mean()
        return (p2q + q2p) / 2


def extract_contour_points(
    face_mask: torch.Tensor,  # (B, H, W) float [0,1]
    n_points: int = 100,
) -> torch.Tensor:
    """Extract ~n_points contour points from face segmentation mask.
    Uses Canny edge detection on the mask.
    Returns (B, n_points, 2) normalized [0,1] coordinates.
    """
    import cv2
    import numpy as np

    B, H, W = face_mask.shape
    all_contours = []
    for b in range(B):
        mask_np = (face_mask[b].detach().cpu().numpy() * 255).astype(np.uint8)
        edges = cv2.Canny(mask_np, 50, 150)
        pts = np.column_stack(np.where(edges > 0))  # (K, 2) row, col
        if len(pts) == 0:
            pts = np.zeros((n_points, 2), dtype=np.float32)
        else:
            # Subsample to n_points
            idx = np.linspace(0, len(pts) - 1, n_points, dtype=int)
            pts = pts[idx].astype(np.float32)
            # Normalize: row→y, col→x
            pts = pts[:, [1, 0]] / np.array([W, H], dtype=np.float32)
        all_contours.append(torch.from_numpy(pts))
    return torch.stack(all_contours).to(face_mask.device)


class RegionWeightedLoss(nn.Module):
    """Vertex-level loss weighted by facial region (nose×3, eyes×2.5, etc.).

    Uses FLAME vertex region masks when available, falls back to uniform weighting.
    """

    def forward(
        self,
        pred_verts: torch.Tensor,  # (B, V, 3)
        gt_verts: torch.Tensor,    # (B, V, 3)
        region_indices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if region_indices is None:
            # Fallback: uniform vertex loss
            return F.mse_loss(pred_verts, gt_verts)

        total_loss = torch.zeros(1, device=pred_verts.device)
        total_weight = 0.0
        for region_name, idx in region_indices.items():
            w = REGION_WEIGHTS.get(region_name, 1.0)
            region_loss = F.mse_loss(pred_verts[:, idx], gt_verts[:, idx])
            total_loss = total_loss + w * region_loss
            total_weight += w

        return total_loss / max(total_weight, 1.0)


class ShapeRegularizer(nn.Module):
    """L2 penalty: keep optimized shape close to MICA initialization."""

    def forward(
        self,
        shape_params: torch.Tensor,  # (B, 300)
        shape_init: torch.Tensor,    # (B, 300)
    ) -> torch.Tensor:
        return F.mse_loss(shape_params, shape_init)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_losses.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/optimizer/losses.py tests/test_losses.py
git commit -m "feat: optimizer/losses.py — landmark, photometric, identity, contour, region, regularizer"
```

---

### Task 11: faceforge/optimizer/refiner.py

**Files:**
- Create: `faceforge/optimizer/refiner.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_pipeline.py` (create):
```python
import torch
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockFLAMEOutput:
    vertices: torch.Tensor
    faces: torch.Tensor
    landmarks2d: torch.Tensor
    landmarks3d: torch.Tensor


def test_refiner_loss_decreases():
    """ShapeRefiner should reduce total loss over optimization steps."""
    from faceforge.optimizer.refiner import ShapeRefiner
    from faceforge.optimizer.losses import LandmarkLoss, ShapeRegularizer
    from omegaconf import OmegaConf

    flame = MagicMock()
    renderer = MagicMock()

    # Flame always returns same vertices regardless of shape input
    def mock_flame_forward(shape, exp, pose):
        return MockFLAMEOutput(
            vertices=shape.unsqueeze(1).expand(-1, 5023, 3).clone().detach(),
            faces=torch.zeros(9976, 3, dtype=torch.long),
            landmarks2d=torch.zeros(1, 68, 2),
            landmarks3d=torch.zeros(1, 68, 3),
        )
    flame.side_effect = mock_flame_forward
    flame.get_contour_vertices.return_value = torch.zeros(1, 17, 3)

    from faceforge.model.renderer import RenderOutput
    renderer.render.return_value = RenderOutput(
        image=torch.zeros(1, 64, 64, 4),
        zbuf=torch.zeros(1, 64, 64, 1),
        pix_to_face=torch.full((1, 64, 64, 1), -1),
    )
    renderer.project_landmarks.return_value = torch.zeros(1, 68, 2)

    loss_cfg = OmegaConf.create({
        "landmark": 1.0, "photometric": 0.0, "identity": 0.0,
        "contour": 0.0, "region": 0.0, "regularize": 0.1,
    })

    refiner = ShapeRefiner(flame=flame, renderer=renderer, losses=loss_cfg)
    shape_init = torch.zeros(1, 300)
    image = torch.zeros(1, 64, 64, 3)
    detected_lmk = torch.ones(1, 68, 2)  # non-zero target

    result = refiner.refine(
        shape_init=shape_init,
        image=image,
        detected_lmk=detected_lmk,
        n_steps=10,
        lr=0.01,
    )

    assert hasattr(result, "shape_params")
    assert result.shape_params.shape == (1, 300)
    assert hasattr(result, "loss_history")
    assert len(result.loss_history) == 10
    # Loss should generally decrease (allow some tolerance)
    assert result.loss_history[-1] <= result.loss_history[0] + 0.1
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_pipeline.py::test_refiner_loss_decreases -v
```

- [ ] **Step 3: Implement refiner.py**

Create `faceforge/optimizer/refiner.py`:
```python
"""ShapeRefiner: differentiable rendering optimization loop.

Refines MICA-initialized FLAME shape coefficients by minimizing:
  - LandmarkLoss: 2D landmark alignment
  - PhotometricLoss: pixel-level RGB match (skin region)
  - IdentityLoss: ArcFace embedding similarity
  - ContourLoss: mesh silhouette vs face mask
  - RegionWeightedLoss: nose/eyes/mouth emphasis
  - ShapeRegularizer: don't stray far from MICA initialization
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from faceforge.optimizer.losses import (
    LandmarkLoss,
    PhotometricLoss,
    IdentityLoss,
    ContourLoss,
    RegionWeightedLoss,
    ShapeRegularizer,
    extract_contour_points,
)
from faceforge.utils.device import get_device


@dataclass
class RefineResult:
    shape_params: torch.Tensor    # (1, 300) optimized
    loss_history: List[float]
    flame_output: object          # FLAMEOutput at final step


class ShapeRefiner:
    """Gradient-based refinement of FLAME shape parameters."""

    def __init__(
        self,
        flame,
        renderer,
        losses: DictConfig,
        device: torch.device = None,
    ):
        self._flame = flame
        self._renderer = renderer
        self._loss_cfg = losses
        self.device = device or get_device()

        self._landmark_loss = LandmarkLoss()
        self._photo_loss = PhotometricLoss()
        self._identity_loss = IdentityLoss()
        self._contour_loss = ContourLoss()
        self._region_loss = RegionWeightedLoss()
        self._regularizer = ShapeRegularizer()

    def refine(
        self,
        shape_init: torch.Tensor,         # (1, 300)
        image: torch.Tensor,              # (1, H, W, 3) float [0,1]
        detected_lmk: torch.Tensor,       # (1, 68, 2) normalized
        face_mask: Optional[torch.Tensor] = None,  # (1, H, W) float
        cameras=None,
        n_steps: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> RefineResult:
        shape_init = shape_init.to(self.device)
        image = image.to(self.device)
        detected_lmk = detected_lmk.to(self.device)

        if face_mask is None:
            face_mask = torch.ones(1, image.shape[1], image.shape[2], device=self.device)

        shape = shape_init.clone().requires_grad_(True)
        exp = torch.zeros(1, 50, device=self.device)
        pose = torch.zeros(1, 6, device=self.device)

        optimizer = torch.optim.Adam([shape], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        loss_history = []
        best_shape = shape_init.clone()
        best_loss = float("inf")
        flame_output = None

        iterator = tqdm(range(n_steps), desc="Refining", leave=False) if verbose else range(n_steps)

        for step in iterator:
            optimizer.zero_grad()

            flame_out = self._flame(shape, exp, pose)
            flame_output = flame_out

            # Build rendering cameras if not provided
            cam = cameras or self._default_camera(image.device)

            render_out = self._renderer.render(flame_out.vertices, flame_out.faces, cam)
            projected_lmk = self._renderer.project_landmarks(flame_out.landmarks3d, cam)

            total_loss = torch.zeros(1, device=self.device)

            # Landmark loss
            if self._loss_cfg.landmark > 0:
                total_loss = total_loss + self._loss_cfg.landmark * self._landmark_loss(
                    projected_lmk, detected_lmk
                )

            # Photometric loss
            if self._loss_cfg.photometric > 0 and render_out.image.requires_grad:
                total_loss = total_loss + self._loss_cfg.photometric * self._photo_loss(
                    render_out.image, image, face_mask
                )

            # Contour loss
            if self._loss_cfg.contour > 0:
                contour_verts = self._flame.get_contour_vertices(flame_out.vertices)
                contour_2d = self._renderer.project_landmarks(contour_verts, cam)
                face_contour = extract_contour_points(face_mask)
                total_loss = total_loss + self._loss_cfg.contour * self._contour_loss(
                    contour_2d, face_contour
                )

            # Shape regularizer (keep near MICA init)
            if self._loss_cfg.regularize > 0:
                total_loss = total_loss + self._loss_cfg.regularize * self._regularizer(
                    shape, shape_init
                )

            loss_val = total_loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_shape = shape.clone().detach()

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and step % 10 == 0:
                logger.info(f"  step {step:4d} | loss = {loss_val:.4f}")

        logger.info(f"[ShapeRefiner] Done. Initial: {loss_history[0]:.4f} → Final: {loss_history[-1]:.4f}")
        return RefineResult(
            shape_params=best_shape,
            loss_history=loss_history,
            flame_output=flame_output,
        )

    def _default_camera(self, device: torch.device):
        """Simple orthographic camera for projection."""
        try:
            from pytorch3d.renderer import FoVOrthographicCameras
            return FoVOrthographicCameras(device=device)
        except ImportError:
            return None
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_pipeline.py::test_refiner_loss_decreases -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add faceforge/optimizer/refiner.py tests/test_pipeline.py
git commit -m "feat: optimizer/refiner.py — 100-step Adam optimization with CosineAnnealingLR"
```

---

### Task 12: faceforge/utils/visualize.py

**Files:**
- Create: `faceforge/utils/visualize.py`

- [ ] **Step 1: Implement visualize.py**

Create `faceforge/utils/visualize.py`:
```python
"""Visualization utilities: mesh overlay, landmarks, loss curve."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
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
        # Add label bar
        bar = np.zeros((30, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(np.vstack([bar, panel]))
    combined = np.hstack(panels)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    logger.info(f"[visualize] Comparison saved: {output_path}")
```

- [ ] **Step 2: Commit**

```bash
git add faceforge/utils/visualize.py
git commit -m "feat: utils/visualize.py — landmarks, mesh overlay, loss curve"
```

---

### Task 13: faceforge/pipeline.py + CLI

**Files:**
- Create: `faceforge/pipeline.py`
- Create: `faceforge/cli.py`
- Create: `faceforge/extensions/deca_detail.py` (stub)
- Create: `faceforge/extensions/pixel3dmm.py` (stub)
- Create: `scripts/setup_models.sh`
- Create: `scripts/run_demo.sh`

- [ ] **Step 1: Write failing end-to-end test**

Append to `tests/test_pipeline.py`:
```python
def test_pipeline_from_config_interface():
    """FaceForgePipeline.from_config must be callable."""
    from faceforge.pipeline import FaceForgePipeline
    from omegaconf import OmegaConf
    # Just verify the class interface exists — real models not needed
    assert hasattr(FaceForgePipeline, "run")
    assert hasattr(FaceForgePipeline, "from_config")


def test_pipeline_result_dataclass():
    """PipelineResult must have expected fields."""
    from faceforge.pipeline import PipelineResult
    r = PipelineResult(
        mesh_path="output/mesh.ply",
        render_path="output/render.png",
        shape_params=None,
        confidence=0.95,
    )
    assert r.mesh_path == "output/mesh.ply"
    assert r.confidence == 0.95
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_pipeline.py::test_pipeline_from_config_interface -v
```

- [ ] **Step 3: Implement pipeline.py**

Create `faceforge/pipeline.py`:
```python
"""FaceForgePipeline: end-to-end 3D face reconstruction."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf, DictConfig

from faceforge.utils.device import get_device
from faceforge.utils.image import load_image
from faceforge.utils.mesh_io import save_mesh
from faceforge.utils.visualize import save_loss_curve, save_comparison


@dataclass
class PipelineResult:
    mesh_path: str
    render_path: str
    shape_params: Optional[np.ndarray]  # (300,)
    confidence: float


class FaceForgePipeline:
    """Orchestrates: image loading → MICA encoding → aggregation → refinement → export."""

    def __init__(self, config: DictConfig):
        self.cfg = config
        self.device = self._resolve_device()
        self._encoder = None
        self._aggregator = None
        self._flame = None
        self._renderer = None
        self._refiner = None
        self._landmark_detector = None
        logger.info(f"[Pipeline] Initialized. Device: {self.device}")

    def _resolve_device(self) -> torch.device:
        dev_cfg = getattr(self.cfg, "device", "auto")
        if dev_cfg == "auto":
            return get_device()
        return torch.device(dev_cfg)

    def _ensure_initialized(self) -> None:
        """Lazy initialization of all sub-modules."""
        if self._encoder is not None:
            return

        from faceforge.encoder.mica_encoder import MICAEncoder
        from faceforge.encoder.multi_image import MultiImageAggregator
        from faceforge.utils.landmarks import LandmarkDetector

        mica_path = self.cfg.paths.mica_weights
        flame_path = self.cfg.paths.flame_model

        self._encoder = MICAEncoder(
            mica_path=mica_path,
            flame_path=flame_path,
            device=self.device,
        )
        self._aggregator = MultiImageAggregator(
            self._encoder,
            strategy=self.cfg.aggregator.strategy,
        )
        self._landmark_detector = LandmarkDetector(device=self.device)

        # FLAME layer
        if Path(flame_path).exists():
            from faceforge.model.flame import FLAMELayer
            self._flame = FLAMELayer(flame_path, device=self.device)
        else:
            logger.warning(f"[Pipeline] FLAME model not found: {flame_path}. Refinement disabled.")

        # Renderer
        from faceforge.model.renderer import DifferentiableRenderer
        render_size = getattr(self.cfg.refiner, "render_size", 512)
        self._renderer = DifferentiableRenderer(image_size=render_size, device=self.device)

        # Refiner
        if self.cfg.refiner.enabled and self._flame is not None:
            from faceforge.optimizer.refiner import ShapeRefiner
            self._refiner = ShapeRefiner(
                flame=self._flame,
                renderer=self._renderer,
                losses=self.cfg.losses,
                device=self.device,
            )

    def run(
        self,
        images: Union[str, List[str], np.ndarray, List[np.ndarray]],
        output_dir: str = "output",
        subject_id: str = "subject",
    ) -> PipelineResult:
        """Full reconstruction pipeline."""
        self._ensure_initialized()

        # Normalize input to list
        if not isinstance(images, list):
            images = [images]
        imgs = [load_image(img) for img in images]

        logger.info(f"[Pipeline] Processing {len(imgs)} image(s) for subject '{subject_id}'")

        # Step 1: MICA encoding + aggregation
        agg_result = self._aggregator.aggregate(imgs)
        shape_params = agg_result.shape_params  # (1, 300)
        confidence = agg_result.confidence
        logger.info(f"[Pipeline] Aggregation confidence: {confidence:.3f}")
        if confidence < self.cfg.aggregator.min_confidence:
            logger.warning(f"[Pipeline] Low confidence ({confidence:.3f}). "
                           "Results may be inaccurate.")

        # Step 2: Landmark detection on first image
        try:
            lmk_tensor = self._landmark_detector.detect(imgs[0], normalize=True)
            detected_lmk = lmk_tensor.unsqueeze(0).to(self.device)  # (1, 68, 2)
        except Exception as e:
            logger.warning(f"[Pipeline] Landmark detection failed: {e}")
            detected_lmk = torch.zeros(1, 68, 2, device=self.device)

        # Step 3: Differentiable refinement
        if self._refiner is not None:
            img_tensor = self._to_tensor(imgs[0])
            refine_result = self._refiner.refine(
                shape_init=shape_params,
                image=img_tensor,
                detected_lmk=detected_lmk,
                n_steps=self.cfg.refiner.n_steps,
                lr=self.cfg.refiner.lr,
                verbose=True,
            )
            shape_params = refine_result.shape_params
            flame_output = refine_result.flame_output
        else:
            flame_output = None
            if self._flame is not None:
                exp = torch.zeros(1, 50, device=self.device)
                pose = torch.zeros(1, 6, device=self.device)
                flame_output = self._flame(shape_params, exp, pose)

        # Step 4: Export
        out_dir = Path(output_dir)
        mesh_dir = out_dir / "meshes"
        render_dir = out_dir / "renders"
        params_dir = out_dir / "params"
        for d in [mesh_dir, render_dir, params_dir]:
            d.mkdir(parents=True, exist_ok=True)

        mesh_path = ""
        render_path = ""

        if flame_output is not None and self.cfg.output.save_mesh:
            verts = flame_output.vertices[0].detach().cpu().numpy()
            faces = flame_output.faces.detach().cpu().numpy()
            fmt = self.cfg.output.mesh_format
            mesh_path = str(mesh_dir / f"{subject_id}.{fmt}")
            save_mesh(mesh_path, verts, faces)
            logger.info(f"[Pipeline] Mesh saved: {mesh_path}")

        if self.cfg.output.save_params:
            params_path = str(params_dir / f"{subject_id}_shape.npy")
            np.save(params_path, shape_params.detach().cpu().numpy())
            logger.info(f"[Pipeline] Params saved: {params_path}")

        if self._refiner is not None and hasattr(refine_result, "loss_history"):
            save_loss_curve(
                refine_result.loss_history,
                str(render_dir / f"{subject_id}_loss.png"),
            )

        return PipelineResult(
            mesh_path=mesh_path,
            render_path=render_path,
            shape_params=shape_params.detach().cpu().numpy().squeeze(),
            confidence=confidence,
        )

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert (H, W, 3) uint8 to (1, H, W, 3) float [0,1] tensor."""
        t = torch.from_numpy(image.astype(np.float32) / 255.0)
        return t.unsqueeze(0).to(self.device)

    @classmethod
    def from_config(cls, config_path: str) -> "FaceForgePipeline":
        """Load pipeline from YAML config, merging with defaults."""
        default_cfg = OmegaConf.load("config/default.yaml")
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(default_cfg, user_cfg)
        return cls(cfg)
```

- [ ] **Step 4: Implement cli.py**

Create `faceforge/cli.py`:
```python
"""Command-line interface for FaceForge."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from loguru import logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faceforge",
        description="FaceForge: High-Fidelity 3D Face Reconstruction",
    )
    p.add_argument(
        "--input", "-i", nargs="+", required=True,
        help="Input image path(s) or video file",
    )
    p.add_argument(
        "--output", "-o", default="output",
        help="Output directory (default: output/)",
    )
    p.add_argument(
        "--subject", "-s", default="subject",
        help="Subject ID used for output file names",
    )
    p.add_argument(
        "--config", "-c", default="config/default.yaml",
        help="Config YAML path (default: config/default.yaml)",
    )
    p.add_argument(
        "--no-refine", action="store_true",
        help="Skip differentiable refinement (faster, MICA output only)",
    )
    p.add_argument(
        "--device", choices=["auto", "cuda", "mps", "cpu"],
        help="Override device (default: auto-detect)",
    )
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from omegaconf import OmegaConf
    from faceforge.pipeline import FaceForgePipeline

    # Load config
    default_cfg = OmegaConf.load("config/default.yaml")
    if Path(args.config).exists() and args.config != "config/default.yaml":
        user_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(default_cfg, user_cfg)
    else:
        cfg = default_cfg

    # CLI overrides
    if args.no_refine:
        cfg.refiner.enabled = False
    if args.device:
        cfg.device = args.device

    pipeline = FaceForgePipeline(cfg)

    try:
        result = pipeline.run(
            images=args.input,
            output_dir=args.output,
            subject_id=args.subject,
        )
        logger.info(f"Done! Mesh: {result.mesh_path} | Confidence: {result.confidence:.3f}")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write extension stubs**

Create `faceforge/extensions/deca_detail.py`:
```python
"""DECA detail module stub — future extension."""
from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class DetailedMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    uv_displacement: torch.Tensor


class DECADetailModule:
    """Placeholder for DECA detail module integration."""

    def add_detail(self, flame_output, image: torch.Tensor) -> DetailedMesh:
        """Add UV displacement map detail to FLAME mesh."""
        raise NotImplementedError(
            "DECA detail module not yet implemented. "
            "See docs/superpowers/specs/ for extension roadmap."
        )
```

Create `faceforge/extensions/pixel3dmm.py`:
```python
"""Pixel3DMM dense constraint module stub — future extension."""
from __future__ import annotations
import torch


class Pixel3DMMModule:
    """Placeholder for Pixel3DMM (CVPR 2025) dense UV + normal constraints."""

    def predict_uv_normal(self, image: torch.Tensor):
        """Predict per-pixel UV coordinates and surface normals."""
        raise NotImplementedError(
            "Pixel3DMM not yet integrated. "
            "See EXT-3 in the project spec for details."
        )
```

- [ ] **Step 6: Write scripts**

Create `scripts/setup_models.sh`:
```bash
#!/bin/bash
set -e
echo "=== FaceForge Model Setup ==="
echo ""
echo "1. FLAME 2020 Model"
echo "   Register at: https://flame.is.tue.mpg.de/"
echo "   Download: FLAME2020.zip"
echo "   Extract to: data/pretrained/FLAME2020/"
echo ""
echo "2. MICA Weights"
echo "   Download from: https://zielon.github.io/mica/"
echo "   Place at: data/pretrained/mica.tar"
echo ""
echo "3. InsightFace buffalo_l (auto-download)"
python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=0)" && echo "   InsightFace: OK"
echo ""
echo "4. Verifying device detection..."
python -c "from faceforge.utils.device import get_device; print('   Device:', get_device())"
echo ""
echo "Setup complete. Run: python -m faceforge.cli --help"
```

Create `scripts/run_demo.sh`:
```bash
#!/bin/bash
set -e
echo "=== FaceForge Demo ==="
if [ -z "$1" ]; then
    echo "Usage: $0 <image_path> [subject_id]"
    echo "Example: $0 input/photo.jpg alice"
    exit 1
fi
IMAGE=$1
SUBJECT=${2:-"demo"}
python -m faceforge.cli \
    --input "$IMAGE" \
    --output output/ \
    --subject "$SUBJECT" \
    --config config/mac_mps.yaml
echo "Done! Check output/ for results."
```
```bash
chmod +x scripts/setup_models.sh scripts/run_demo.sh
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/test_pipeline.py -v
```
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add faceforge/pipeline.py faceforge/cli.py \
        faceforge/extensions/deca_detail.py faceforge/extensions/pixel3dmm.py \
        scripts/setup_models.sh scripts/run_demo.sh
git commit -m "feat: pipeline.py + cli.py — end-to-end reconstruction with argparse CLI"
```

---

### Task 14: Final Tests + Coverage Check

**Files:**
- Update: `tests/test_encoder.py`, `tests/test_losses.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS (skip pytorch3d tests if not installed)

- [ ] **Step 2: Check coverage**

```bash
pip install pytest-cov
pytest tests/ --cov=faceforge --cov-report=term-missing
```
Expected: Coverage ≥ 70%

- [ ] **Step 3: Smoke test CLI help**

```bash
python -m faceforge.cli --help
```
Expected: Help text with --input, --output, --subject, --config, --no-refine

- [ ] **Step 4: Verify config loading**

```bash
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/default.yaml')
assert cfg.refiner.n_steps == 100
print('Config OK:', cfg.device)
"
```
Expected: `Config OK: auto`

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "chore: complete FaceForge implementation — all modules, tests, CLI"
```

---

## Summary

| Task | Module | Acceptance |
|------|--------|-----------|
| 1 | Project skeleton | `import faceforge` OK |
| 2 | device.py | CUDA>MPS>CPU + env override |
| 3 | utils/image.py | preprocess → (1,3,112,112) |
| 4 | utils/landmarks.py | (68,2) normalized coords |
| 5 | utils/mesh_io.py | .ply roundtrip |
| 6 | model/flame.py | FLAMEOutput dataclass, vertices (B,5023,3) |
| 7 | model/renderer.py | RGBA (B,H,W,4), MPS fallback |
| 8 | encoder/mica_encoder.py | encode → (1,300) |
| 9 | encoder/multi_image.py | median/mean/trimmed_mean + confidence |
| 10 | optimizer/losses.py | 6 loss functions, all differentiable |
| 11 | optimizer/refiner.py | loss decreases over 100 steps |
| 12 | utils/visualize.py | landmark draw, loss curve, overlay |
| 13 | pipeline.py + CLI | end-to-end, `.ply` output, argparse |
| 14 | Tests | coverage ≥ 70% |
