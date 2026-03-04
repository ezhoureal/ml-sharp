"""Palettize the SHARP model to 4-bit and convert to Core ML format.

This script applies post-training palettization to the SHARP model,
quantizing weights to 4 bits (16 palette levels) for efficient
deployment on Apple devices.

The palettized model is converted directly to Core ML format (.mlpackage).

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import coremltools as ct
import torch
from coremltools.optimize.torch.palettization import (
    PostTrainingPalettizer,
    PostTrainingPalettizerConfig,
)

from sharp.models import PredictorParams, create_predictor

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_RESOLUTION = (1536, 1536)


def create_calibration_data(
    device: torch.device,
    num_samples: int = 8,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create calibration data for palettization.

    Args:
        device: The device to run on.
        num_samples: Number of random calibration samples.
        resolution: Image resolution for calibration.

    Returns:
        List of (image, disparity_factor) tuples.
    """
    height, width = resolution
    calibration_data = []

    LOGGER.info(f"Creating {num_samples} calibration samples...")
    for _ in range(num_samples):
        # Create random RGB image
        image = torch.rand(1, 3, height, width, device=device)

        # Create random disparity factor (typical focal length values)
        f_px = torch.rand(1, device=device) * 500 + 500  # Between 500-1000
        disparity_factor = f_px / width

        calibration_data.append((image, disparity_factor))

    return calibration_data


def palettize_and_convert(
    model_path: Path | None = None,
    output_path: Path | None = None,
    device: str = "default",
    n_bits: int = 4,
) -> Path:
    """Palettize the SHARP model and convert to Core ML format.

    Args:
        model_path: Path to the model checkpoint. If None, downloads default model.
        output_path: Path to save the Core ML model.
        device: Device to run on ('cpu', 'cuda', or 'mps').
        n_bits: Number of bits for palettization (default 4 = 16 levels).
        minimum_deployment_target: Minimum deployment target for Core ML conversion.

    Returns:
        Path to the saved Core ML model.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Starting SHARP model palettization and Core ML conversion")

    # Determine device
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Force CPU for MPS since coremltools K-means doesn't support it
    if device == "mps":
        device = "cpu"
        LOGGER.info("MPS detected but using CPU (coremltools K-means requires CPU)")

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

    # Configure palettization with per_tensor granularity for better compression
    palettization_config_dict = {
        "global_config": {
            "n_bits": n_bits,
            "granularity": "per_tensor",
            "lut_dtype": torch.int8,  # Use int8 for LUT indices to save memory
        },
    }

    LOGGER.info(f"Palettization config: {palettization_config_dict}")
    LOGGER.info(f"Target: {n_bits}-bit palettization ({2**n_bits} levels per LUT)")

    # Create palettizer
    palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
    palettizer = PostTrainingPalettizer(model, palettization_config)

    # Apply palettization
    LOGGER.info("Applying post-training palettization...")

    # Log skipped layers before compression
    skipped_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight_shape = cast(torch.Size, module.weight.shape)
            # Check if shape would cause issues with palettization
            if len(weight_shape) < 2:
                skipped_layers.append((name, weight_shape, "insufficient dimensions"))
            elif weight_shape[0] == 1 or weight_shape[1] == 1:
                skipped_layers.append((name, weight_shape, "dimension size of 1"))

    if skipped_layers:
        LOGGER.info(f"Layers that may be skipped during palettization:")
        for name, shape, reason in skipped_layers:
            LOGGER.info(f"  - {name}: shape {shape}, reason: {reason}")

    # Compress the model using K-means clustering on weights
    palettized_model = palettizer.compress()

    LOGGER.info("Palettization complete!")

    # Count palettized modules
    palettized_modules = 0
    for module in palettized_model.modules():
        if hasattr(module, "_is_palettized") and module._is_palettized:
            palettized_modules += 1

    LOGGER.info(f"Number of palettized modules: {palettized_modules}")

    # Determine output path
    if output_path is None:
        output_path = Path(f"sharp_palettized_{n_bits}bit.mlpackage")

    # Ensure we're saving as .mlpackage
    if output_path.suffix != ".mlpackage":
        output_path = output_path.with_suffix(".mlpackage")

    LOGGER.info(f"Exporting to Core ML format ({output_path})...")

    # Trace the model to TorchScript
    LOGGER.info("Tracing model to TorchScript...")
    example_image = torch.rand(
        1, 3, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1], device=device_obj
    )
    example_disparity = torch.tensor([0.5], device=device_obj)

    with torch.no_grad():
        traced_model = torch.jit.trace(palettized_model, (example_image, example_disparity))

    # Export to Core ML format
    LOGGER.info("Converting to Core ML...")
    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[
            ct.TensorType(
                name="image",
                shape=(1, 3, DEFAULT_RESOLUTION[0], DEFAULT_RESOLUTION[1]),
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

def compare_outputs(
    original_model: torch.nn.Module,
    palettized_model: torch.nn.Module,
    device: torch.device,
    num_samples: int = 5,
) -> None:
    """Compare outputs of original and palettized models.

    Args:
        original_model: The original model.
        palettized_model: The palettized model.
        device: Device to run on.
        num_samples: Number of test samples.
    """
    LOGGER.info("Comparing outputs of original vs palettized model...")
    original_model.eval()
    palettized_model.eval()

    height, width = DEFAULT_RESOLUTION
    max_diff = 0.0
    total_diff = 0.0

    for _ in range(num_samples):
        with torch.no_grad():
            # Create random test input
            image = torch.rand(1, 3, height, width, device=device)
            f_px = torch.tensor([800.0], device=device)
            disparity_factor = f_px / width

            # Get outputs
            original_output = original_model(image, disparity_factor)
            palettized_output = palettized_model(image, disparity_factor)

            # Compare outputs - handle both dataclass and dict outputs
            def get_tensors(output):
                if isinstance(output, dict):
                    return output.values()
                elif hasattr(output, "__dataclass_fields__"):
                    return [getattr(output, k) for k in output.__dataclass_fields__]
                elif isinstance(output, (list, tuple)):
                    return output
                else:
                    return [output]

            orig_tensors = get_tensors(original_output)
            palett_tensors = get_tensors(palettized_output)

            for orig_tensor, palett_tensor in zip(orig_tensors, palett_tensors):
                if torch.is_tensor(orig_tensor) and torch.is_tensor(palett_tensor):
                    diff = (orig_tensor - palett_tensor).abs().max().item()
                    max_diff = max(max_diff, diff)
                    total_diff += diff

    LOGGER.info(f"Maximum absolute difference: {max_diff:.6f}")
    LOGGER.info(f"Average maximum difference: {total_diff / num_samples:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Palettize SHARP model to 4-bit and convert to Core ML format"
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
        help="Path to save the Core ML model (default: sharp_palettized_4bit.mlpackage).",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=4,
        help="Number of bits for palettization (default: 4 = 16 levels).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare outputs of original and palettized models.",
    )

    args = parser.parse_args()

    # Palettize and convert
    output_path = palettize_and_convert(
        model_path=args.checkpoint_path,
        output_path=args.output_path,
        n_bits=args.n_bits,
    )

    # Compare outputs if requested
    if args.compare:
        device_obj = torch.device("cpu")

        # Load original model for comparison
        original_model = create_predictor(PredictorParams())
        if args.checkpoint_path is None:
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        else:
            state_dict = torch.load(args.checkpoint_path, weights_only=True)
        original_model.load_state_dict(state_dict)
        original_model.eval()
        original_model.to(device_obj)

        # Load the palettized model from the saved Core ML package for comparison
        # We need to re-create and re-palettize since we don't save the .pt anymore
        LOGGER.info("Reloading palettized model for comparison...")

        # Re-create the palettized model
        palettization_config_dict = {
            "global_config": {
                "n_bits": args.n_bits,
                "granularity": "per_tensor",
            },
        }
        palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
        palettizer = PostTrainingPalettizer(original_model, palettization_config)
        palettized_model = palettizer.compress()

        compare_outputs(original_model, palettized_model, device_obj)
