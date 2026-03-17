__version__ = "1.0.0"

def __getattr__(name):
    if name == "FaceForgePipeline":
        from faceforge.pipeline import FaceForgePipeline
        return FaceForgePipeline
    raise AttributeError(f"module 'faceforge' has no attribute {name!r}")

__all__ = ["FaceForgePipeline"]
