"""
Tests for faceforge.pipeline — placeholder until pipeline is implemented.
"""
import pytest


def test_pipeline_module_importable():
    """pipeline.py stub must be importable without errors."""
    import faceforge.pipeline  # noqa: F401


def test_faceforge_version():
    """faceforge.__version__ must be set."""
    import faceforge
    assert faceforge.__version__ == "1.0.0"
