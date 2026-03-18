"""FaceForge CLI — typer-based command-line interface."""
import typer
from typing import List, Optional
from pathlib import Path
from loguru import logger

app = typer.Typer(help="FaceForge - High-Fidelity 3D Face Reconstruction")


@app.command()
def reconstruct(
    input: List[Path] = typer.Option(..., "--input", "-i",
        help="Input image path(s). Pass multiple -i flags for multi-image mode."),
    output: Path = typer.Option(Path("output"), "--output", "-o",
        help="Output directory"),
    subject: str = typer.Option("subject", "--subject", "-s",
        help="Subject ID used for output file names"),
    config: Path = typer.Option(Path("config/default.yaml"), "--config", "-c",
        help="Config YAML path"),
    no_refine: bool = typer.Option(False, "--no-refine",
        help="Skip differentiable refinement (fast mode, MICA output only)"),
    save_intermediates: bool = typer.Option(False, "--save-intermediates",
        help="Save per-stage intermediate artifacts (aligned crops, mesh previews, etc.)"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
        help="Print optimization progress"),
):
    """Reconstruct a 3D face mesh from one or more input images."""
    from omegaconf import OmegaConf
    from faceforge.pipeline import FaceForgePipeline

    # Load and merge configs
    default_cfg = OmegaConf.load("config/default.yaml")
    if config.exists() and str(config) != "config/default.yaml":
        override_cfg = OmegaConf.load(str(config))
        cfg = OmegaConf.merge(default_cfg, override_cfg)
    else:
        cfg = default_cfg

    # Apply CLI overrides
    if no_refine:
        cfg.refiner.enabled = False
    if save_intermediates:
        cfg.output.save_intermediates = True

    logger.info(f"[CLI] Input: {[str(p) for p in input]}")
    logger.info(f"[CLI] Output: {output} | Subject: {subject}")

    pipeline = FaceForgePipeline(cfg)
    result = pipeline.run(
        images=[str(p) for p in input],
        output_dir=str(output),
        subject_id=subject,
    )

    typer.echo(f"\n✓ Reconstruction complete!")
    typer.echo(f"  Mesh:       {result.mesh_path or 'not saved'}")
    typer.echo(f"  Params:     {result.params_path or 'not saved'}")
    typer.echo(f"  Confidence: {result.confidence:.3f}")
    if result.loss_final > 0:
        typer.echo(f"  Final loss: {result.loss_final:.4f}")


if __name__ == "__main__":
    app()
