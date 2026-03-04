"""Convert the original SHARP model to Core ML format (FP32, no quantization).

This script converts the original full-precision PyTorch model to Core ML format
without applying palettization, for comparison with the quantized version.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coremltools as ct
import torch

from sharp.models import PredictorParams, create_predictor

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_RESOLUTION = (1536, 1536)


def convert_to_coreml(
    model_path: Path | None = None,
    output_path: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Convert the original SHARP model to Core ML format (FP32).

    Args:
        model_path: Path to the model checkpoint. If None, downloads default model.
        output_path: Path to save the Core ML model.
        device: Device to run on ('cpu', 'cuda', or 'mps').

    Returns:
        Path to the saved Core ML model.
    """
    LOGGER.info("Starting SHARP model conversion to Core ML (FP32)")

    # Determine device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_obj = torch.device(device)
    LOGGER.info(f"Using device: {device}")

    # Load model
    if model_path is None:
        LOGGER.info(f"Downloading default model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, weights_only=True)

    LOGGER.info("Creating model architecture...")
    model = create_predictor(PredictorParams())
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device_obj)

    # Determine output path
    if output_path is None:
        output_path = Path("sharp_fp32.mlpackage")

    # Ensure we're saving as .mlpackage
    if output_path.suffix != ".mlpackage":
        output_path = output_path.with_suffix(".mlpackage")

    LOGGER.info(f"Exporting to Core ML format ({output_path})...")

    # Trace the model to TorchScript
    LOGGER.info("Tracing model to TorchScript...")
    example_image = torch.rand(
        1, 3, DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[0], device=device_obj
    )
    example_disparity = torch.tensor([0.5], device=device_obj)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, (example_image, example_disparity))

    # Export to Core ML format
    LOGGER.info("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[
            ct.TensorType(
                name="image",
                shape=(1, 3, DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[0]),
            ),
            ct.TensorType(name="disparity_factor", shape=(1,)),
        ],
        minimum_deployment_target=ct.target.macOS26,
    )

    if not isinstance(mlmodel, ct.models.MLModel):
        LOGGER.error("Failed to convert model to Core ML format.")
        raise RuntimeError("Core ML conversion failed.")

    # Save the model
    mlmodel.save(str(output_path))
    LOGGER.info(f"Core ML model saved to {output_path}")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert original SHARP model to Core ML (FP32, no quantization)"
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to the .pt checkpoint. If not provided, downloads the default model.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help="Path to save the Core ML model (default: sharp_fp32.mlpackage).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    convert_to_coreml(
        model_path=args.checkpoint_path,
        output_path=args.output_path,
    )

    return 0


if __name__ == "__main__":
    exit(main())
