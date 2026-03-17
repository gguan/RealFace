"""
Shared pytest fixtures for the FaceForge test suite.
"""
import numpy as np
import pytest


@pytest.fixture
def random_face_image():
    """Return a random (256, 256, 3) uint8 RGB image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def simple_mesh():
    """Return a minimal valid mesh: a single tetrahedron."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int32)
    return vertices, faces
