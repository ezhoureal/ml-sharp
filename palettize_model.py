"""Palettize the SHARP model to 4-bit using coremltools.

This script applies post-training palettization to the SHARP model,
quantizing weights to 4 bits (16 palette levels) for efficient
deployment on Apple devices.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from coremltools.optimize.torch.palettization import (
    PostTrainingPalettizer,
    PostTrainingPalettizerConfig,
)

from sharp.models import PredictorParams, create_predictor

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
DEFAULT_RESOLUTION = (1536, 1536)


def create_calibration_data(
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int = 8,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create calibration data for palettization.

    Args:
        model: The model to generate calibration data for.
        device: The device to run on.
        num_samples: Number of random calibration samples.
        resolution: Image resolution for calibration.

    Returns:
        List of (image, disparity_factor) tuples.
    """
    height, width = resolution
    calibration_data = []

    LOGGER.info(f"Creating {num_samples} calibration samples...")
    for i in range(num_samples):
        # Create random RGB image
        image = torch.rand(1, 3, height, width, device=device)

        # Create random disparity factor (typical focal length values)
        f_px = torch.rand(1, device=device) * 500 + 500  # Between 500-1000
        disparity_factor = f_px / width

        calibration_data.append((image, disparity_factor))

    return calibration_data


def palettize_model(
    model_path: Path | None = None,
    output_path: Path | None = None,
    n_bits: int = 4,
    granularity: str = "per_grouped_channel",
    group_size: int = 4,
    calibration_nsamples: int = 8,
    device: str = "cpu",
) -> torch.nn.Module:
    """Palettize the SHARP model to n-bits.

    Args:
        model_path: Path to the model checkpoint. If None, downloads default model.
        output_path: Path to save the palettized model.
        n_bits: Number of bits for palettization (default 4 = 16 levels).
        granularity: Granularity for palettization ('per_tensor' or 'per_grouped_channel').
        group_size: Number of channels in a group for per_grouped_channel.
        calibration_nsamples: Number of calibration samples.
        device: Device to run on ('cpu', 'cuda', or 'mps').

    Returns:
        The palettized model.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Starting SHARP model palettization")

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

    # Configure palettization
    palettization_config_dict = {
        "global_config": {
            "n_bits": n_bits,
            "granularity": granularity,
            "group_size": group_size,
            "lut_dtype": "int8",
        },
    }

    LOGGER.info(f"Palettization config: {palettization_config_dict}")
    LOGGER.info(f"Target: {n_bits}-bit palettization ({2**n_bits} levels per LUT)")

    # Create palettizer
    palettization_config = PostTrainingPalettizerConfig.from_dict(palettization_config_dict)
    palettizer = PostTrainingPalettizer(model, palettization_config)

    # Prepare calibration data
    calibration_data = create_calibration_data(model, device_obj, calibration_nsamples)

    # Apply palettization
    LOGGER.info("Applying post-training palettization...")
    palettized_model = palettizer.compress()

    LOGGER.info("Palettization complete!")

    # Save palettized model
    if output_path is None:
        output_path = Path("sharp_palettized_4bit.pt")

    LOGGER.info(f"Saving palettized model to {output_path}")
    torch.save(
        {"model_state_dict": palettized_model.state_dict(), "config": palettization_config_dict},
        output_path,
    )

    LOGGER.info(f"Palettized model saved to {output_path}")

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    palettized_params = sum(p.numel() for p in palettized_model.parameters())

    LOGGER.info(f"Original parameters: {total_params:,}")
    LOGGER.info(f"Palettized parameters: {palettized_params:,}")

    # Count palettized modules
    palettized_modules = 0
    for module in palettized_model.modules():
        if hasattr(module, "_is_palettized") and module._is_palettized:
            palettized_modules += 1

    LOGGER.info(f"Number of palettized modules: {palettized_modules}")

    return palettized_model


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

    for i in range(num_samples):
        with torch.no_grad():
            # Create random test input
            image = torch.rand(1, 3, height, width, device=device)
            f_px = torch.tensor([800.0], device=device)
            disparity_factor = f_px / width

            # Get outputs
            original_output = original_model(image, disparity_factor)
            palettized_output = palettized_model(image, disparity_factor)

            # Compare gaussians output
            for key in original_output:
                orig_tensor = getattr(original_output, key)
                palett_tensor = getattr(palettized_output, key)

                if torch.is_tensor(orig_tensor) and torch.is_tensor(palett_tensor):
                    diff = (orig_tensor - palett_tensor).abs().max().item()
                    max_diff = max(max_diff, diff)
                    total_diff += diff

    LOGGER.info(f"Maximum absolute difference: {max_diff:.6f}")
    LOGGER.info(f"Average maximum difference: {total_diff / num_samples:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Palettize SHARP model to 4-bit using coremltools"
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
        help="Path to save the palettized model.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=4,
        help="Number of bits for palettization (default: 4 = 16 levels).",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="per_grouped_channel",
        choices=["per_tensor", "per_grouped_channel"],
        help="Granularity for palettization.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Group size for per_grouped_channel granularity.",
    )
    parser.add_argument(
        "--calibration-nsamples",
        type=int,
        default=8,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps", "default"],
        help="Device to run on.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare outputs of original and palettized models.",
    )

    args = parser.parse_args()

    # Set default output path
    if args.output_path is None:
        args.output_path = Path(f"sharp_palettized_{args.n_bits}bit.pt")

    # Palettize model
    original_model = create_predictor(PredictorParams())

    # Load original model for comparison
    if args.compare:
        if args.checkpoint_path is None:
            state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        else:
            state_dict = torch.load(args.checkpoint_path, weights_only=True)
        original_model.load_state_dict(state_dict)
        original_model.eval()
        original_model.to(args.device)

    # Palettize
    palettized_model = palettize_model(
        model_path=args.checkpoint_path,
        output_path=args.output_path,
        n_bits=args.n_bits,
        granularity=args.granularity,
        group_size=args.group_size,
        calibration_nsamples=args.calibration_nsamples,
        device=args.device,
    )

    # Compare outputs if requested
    if args.compare:
        compare_outputs(original_model, palettized_model, torch.device(args.device))
