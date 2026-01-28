"""Cloud GPU operations via Modal."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import click

from sharp.utils import io
from sharp.utils import logging as logging_utils

LOGGER = logging.getLogger(__name__)

GpuTier = Literal["t4", "l4", "a10", "a100", "h100"]


def check_modal_installed() -> bool:
    """Check if modal is installed."""
    try:
        import modal  # noqa: F401

        return True
    except ImportError:
        return False


@click.group()
def cloud_cli():
    """Cloud GPU operations via Modal.

    Run SHARP inference on Modal's cloud GPUs. Requires Modal to be installed
    and configured with `modal token new`.

    Install with: pip install -e ".[cloud]"
    """
    pass


@cloud_cli.command("predict")
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or directory containing images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians (PLY/SPLAT/SOG files).",
    required=True,
)
@click.option(
    "-f",
    "--format",
    "export_formats",
    type=click.Choice(["ply", "splat", "sog"], case_sensitive=False),
    multiple=True,
    default=["ply"],
    help="Output format(s). Can specify multiple: -f ply -f splat -f sog",
)
@click.option(
    "--gpu",
    type=click.Choice(["t4", "l4", "a10", "a100", "h100"]),
    default="a10",
    help="GPU tier. Prices/hr: t4=$0.59, l4=$0.80, a10=$1.10, a100=$2.50, h100=$3.95.",
)
@click.option(
    "--batch-size",
    type=int,
    default=20,
    show_default=True,
    help="Number of images per cloud batch to avoid Modal timeouts.",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def cloud_predict_cli(
    input_path: Path,
    output_path: Path,
    export_formats: tuple[str, ...],
    gpu: GpuTier,
    batch_size: int,
    verbose: bool,
):
    """Predict Gaussians from input images using Modal cloud GPUs.

    This command uploads your images to Modal's cloud infrastructure,
    runs inference on the specified GPU tier, and downloads the resulting
    files to your local machine.

    Examples:
        sharp cloud predict -i photo.jpg -o output/

        sharp cloud predict -i photos/ -o output/ --gpu a100
    """
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if batch_size < 1:
        raise click.BadParameter("batch-size must be >= 1")
    if batch_size > 20:
        LOGGER.warning("batch-size capped at 20 (requested %d)", batch_size)
        batch_size = 20

    # Normalize export formats to lowercase
    export_formats = tuple(fmt.lower() for fmt in export_formats)

    if not check_modal_installed():
        click.echo(
            "Modal is not installed. Install it with:\n"
            "  pip install -e '.[cloud]'\n"
            "  # or\n"
            "  uv pip install -e '.[cloud]'\n\n"
            "Then configure Modal with:\n"
            "  modal token new",
            err=True,
        )
        raise SystemExit(1)

    # Import Modal components (only after checking installation)
    from sharp.modal.app import app, get_predict_function

    # Find images to process
    extensions = {ext.lower() for ext in io.get_supported_image_extensions()}
    image_paths: list[Path] = []

    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            image_paths = [input_path]
    else:
        seen = set()
        for candidate_path in input_path.rglob("*"):
            if not candidate_path.is_file():
                continue
            if candidate_path.suffix.lower() not in extensions:
                continue

            resolved = candidate_path.resolve()
            if resolved in seen:
                continue

            seen.add(resolved)
            image_paths.append(candidate_path)
        image_paths.sort()

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d image(s) on Modal with GPU: %s", len(image_paths), gpu)

    # Create output directory
    output_path.mkdir(exist_ok=True, parents=True)

    # Get the appropriate function for the GPU tier
    predict_fn = get_predict_function(gpu)

    # Process batches within Modal app context
    with app.run():
        total = len(image_paths)
        for batch_index in range(0, total, batch_size):
            batch_paths = image_paths[batch_index : batch_index + batch_size]
            batch_number = batch_index // batch_size + 1
            batch_count = (total + batch_size - 1) // batch_size

            LOGGER.info(
                "Uploading batch %d/%d with %d image(s)",
                batch_number,
                batch_count,
                len(batch_paths),
            )
            image_batch = [
                (image_path.read_bytes(), image_path.name) for image_path in batch_paths
            ]

            try:
                outputs = predict_fn.remote(
                    image_batch=image_batch,
                    export_formats=export_formats,
                )

                for output_filename, output_bytes in outputs:
                    output_file = output_path / output_filename
                    output_file.write_bytes(output_bytes)
                    LOGGER.info("Saved %s", output_file)

            except Exception as e:
                LOGGER.error("Failed to process batch %d/%d: %s", batch_number, batch_count, e)
                raise

    LOGGER.info("Done! Processed %d image(s).", len(image_paths))


@cloud_cli.command("setup")
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def cloud_setup_cli(verbose: bool):
    """Set up Modal and pre-download the model.

    This command provisions the Modal volume and downloads the SHARP model
    to the cloud cache. Run this once to ensure fast inference on subsequent calls.
    """
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not check_modal_installed():
        click.echo(
            "Modal is not installed. Install it with:\n"
            "  pip install -e '.[cloud]'\n"
            "  # or\n"
            "  uv pip install -e '.[cloud]'\n\n"
            "Then configure Modal with:\n"
            "  modal token new",
            err=True,
        )
        raise SystemExit(1)

    LOGGER.info("Setting up Modal for SHARP inference...")

    # Import and trigger model download by calling a test function
    # Create a minimal test image (1x1 white pixel PNG)
    import io as stdlib_io

    from PIL import Image

    from sharp.modal.app import app, predict_gaussian_splat_a10

    img = Image.new("RGB", (100, 100), color="white")
    buffer = stdlib_io.BytesIO()
    img.save(buffer, format="PNG")
    test_bytes = buffer.getvalue()

    LOGGER.info("Warming up Modal function and downloading model...")

    try:
        with app.run():
            predict_gaussian_splat_a10.remote(image_batch=[(test_bytes, "test.png")])
        LOGGER.info("Setup complete! Model is cached and ready for inference.")
    except Exception as e:
        error_message = str(e)
        LOGGER.error("Setup failed: %s", error_message)

        # Provide helpful error-specific guidance
        if "token" in error_message.lower() or "auth" in error_message.lower():
            LOGGER.info(
                "Make sure you have configured Modal with 'modal token new' "
                "and have sufficient credits."
            )
        elif "zip archive" in error_message.lower() or "corrupted" in error_message.lower():
            LOGGER.info(
                "The model cache appears corrupted. Please try running setup again - "
                "it will automatically re-download the model."
            )
        else:
            LOGGER.info("Check your Modal configuration and network connection.")

        raise SystemExit(1)
