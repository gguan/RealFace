"""Mesh I/O utilities: save and load PLY/OBJ meshes via trimesh."""

import numpy as np
import trimesh
from pathlib import Path


def save_mesh(
    vertices: np.ndarray,  # (V, 3)
    faces: np.ndarray,     # (F, 3)
    path: str,
    format: str = "ply",   # "ply" | "obj"
) -> str:
    """Save mesh, return actual saved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(str(path))
    return str(path)


def load_mesh(path: str) -> tuple:
    """Returns (vertices, faces) as numpy arrays."""
    mesh = trimesh.load(str(path), process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces
