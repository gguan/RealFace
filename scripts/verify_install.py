#!/usr/bin/env python3
"""
verify_install.py — Sanity-check that FaceForge dependencies are importable
and that device detection works correctly.

Run from repository root:
    python scripts/verify_install.py
"""

import sys


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [OK]  {label}: {result}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")
        return False


failures = 0

print("=== FaceForge install verification ===\n")

# --- Core imports ---
print("-- Core dependencies --")
failures += not check("torch", lambda: __import__("torch").__version__)
failures += not check("torchvision", lambda: __import__("torchvision").__version__)
failures += not check("numpy", lambda: __import__("numpy").__version__)
failures += not check("scipy", lambda: __import__("scipy").__version__)

print("\n-- 3D / Rendering --")
failures += not check("trimesh", lambda: __import__("trimesh").__version__)

print("\n-- Face detection / landmarks --")
failures += not check("insightface", lambda: __import__("insightface").__version__)
failures += not check("onnxruntime", lambda: __import__("onnxruntime").__version__)
failures += not check("face_alignment", lambda: __import__("face_alignment").__version__)

print("\n-- Image utilities --")
failures += not check("cv2 (opencv)", lambda: __import__("cv2").__version__)
failures += not check("PIL (Pillow)", lambda: __import__("PIL").__version__)

print("\n-- Config / Utils --")
failures += not check("omegaconf", lambda: __import__("omegaconf").__version__)
failures += not check("loguru", lambda: __import__("loguru").__version__)
failures += not check("tqdm", lambda: __import__("tqdm").__version__)
failures += not check("rich", lambda: __import__("rich").__version__)

print("\n-- FaceForge package --")
failures += not check("faceforge", lambda: __import__("faceforge").__version__)

print("\n-- Device detection --")

def _device_check():
    from faceforge.utils.device import get_device
    device = get_device()
    return str(device)

failures += not check("get_device()", _device_check)

# --- Config loading ---
print("\n-- Config loading --")

def _config_check():
    from omegaconf import OmegaConf
    import pathlib
    cfg_path = pathlib.Path(__file__).parent.parent / "config" / "default.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    return f"device={cfg.device}, refiner.n_steps={cfg.refiner.n_steps}"

failures += not check("config/default.yaml", _config_check)

print(f"\n=== Result: {failures} failure(s) ===")
sys.exit(0 if failures == 0 else 1)
