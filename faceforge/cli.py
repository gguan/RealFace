"""FaceForge CLI — full implementation in Phase 5."""
import typer
from typing import List, Optional
from pathlib import Path

app = typer.Typer(help="FaceForge - High-Fidelity 3D Face Reconstruction")


@app.command()
def reconstruct(
    input: List[Path] = typer.Option(..., "--input", "-i", help="Input image path(s)"),
    output: Path = typer.Option("output", "--output", "-o"),
    subject: str = typer.Option("subject", "--subject", "-s"),
    config: Path = typer.Option("config/default.yaml", "--config", "-c"),
    no_refine: bool = typer.Option(False, "--no-refine", help="Skip refinement (fast mode)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Reconstruct a 3D face mesh from one or more input images."""
    raise NotImplementedError("CLI not yet implemented — coming in Phase 5")


if __name__ == "__main__":
    app()
