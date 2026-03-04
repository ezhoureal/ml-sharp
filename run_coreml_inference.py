"""Run inference on the Core ML palettized model.

Example usage:
    uv run python run_coreml_inference.py -i data/sample.jpg -o output.ply

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_RESOLUTION = (1536, 1536)


def load_coreml_model(model_path: Path) -> ct.models.MLModel:
    """Load the Core ML model.

    Args:
        model_path: Path to the .mlpackage file.

    Returns:
        Loaded Core ML model.
    """
    LOGGER.info(f"Loading Core ML model from {model_path}")
    model = ct.models.MLModel(str(model_path))
    return model


def preprocess_image(
    image: np.ndarray, target_size: tuple[int, int] = DEFAULT_RESOLUTION
) -> np.ndarray:
    """Preprocess image for Core ML inference.

    Args:
        image: Input image as numpy array (H, W, C) in range [0, 255].
        target_size: Target resolution (width, height).

    Returns:
        Preprocessed image as numpy array (1, 3, H, W).
    """
    # Convert to torch tensor for resizing
    image_pt = (
        torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
    )  # (C, H, W)

    # Resize to target resolution
    image_resized = F.interpolate(
        image_pt[None],  # Add batch dim
        size=(target_size[1], target_size[0]),  # (H, W)
        mode="bilinear",
        align_corners=True,
    )

    # Convert back to numpy for Core ML
    return image_resized.numpy()


def run_inference(
    model: ct.models.MLModel | ct.models.CompiledMLModel,
    image: np.ndarray,
    f_px: float,
) -> dict[str, np.ndarray]:
    """Run inference on a single image.

    Args:
        model: Core ML model.
        image: Input image as numpy array (H, W, C) in range [0, 255].
        f_px: Focal length in pixels.

    Returns:
        Dictionary of output tensors.
    """
    height, width = image.shape[:2]
    # Calculate disparity_factor using original image width (like original predict_image)
    # The disparity_factor = f_px / original_width
    disparity_factor = f_px / width

    # Preprocess image
    LOGGER.info("Preprocessing image...")
    image_input = preprocess_image(image)  # (1, 3, 1536, 1536)

    # Prepare inputs for Core ML
    inputs = {
        "image": image_input,
        "disparity_factor": np.array([disparity_factor], dtype=np.float32),
    }

    # Run inference
    LOGGER.info("Running Core ML inference...")
    predictions = model.predict(inputs)

    return predictions


def postprocess_outputs(
    predictions: dict[str, np.ndarray],
    f_px: float,
    original_size: tuple[int, int],
    internal_size: tuple[int, int] = DEFAULT_RESOLUTION,
) -> Gaussians3D:
    """Convert Core ML outputs to Gaussians3D.

    Args:
        predictions: Dictionary of output tensors from Core ML.
        f_px: Focal length in pixels.
        original_size: Original image size (width, height).
        internal_size: Internal resolution used by model (width, height).

    Returns:
        Gaussians3D object.
    """
    LOGGER.info("Postprocessing outputs...")

    device = torch.device("cpu")

    # Convert predictions to torch tensors (keep batch dim)
    # The Core ML model returns generic var_XXXX names, map by shape
    pred_torch: dict[str, torch.Tensor] = {}
    for k, v in predictions.items():
        tensor = torch.from_numpy(v).to(device)  # Keep batch dim: (1, N, ...)
        pred_torch[k] = tensor
        LOGGER.debug(f"  {k}: shape {tensor.shape}")

    # Map outputs by shape and value ranges to Gaussians3D fields
    # Based on value range analysis:
    # var_5461: [-1.4, 1.3] -> mean_vectors (positions, can be negative)
    # var_5465: [0, 0.03] -> singular_values (scales, small positive)
    # var_5453: [-30, 4] -> quaternions (any value)
    # var_5456: [0, 0.99] -> colors (normalized 0-1)
    # var_5457: [0, 1] -> opacities (0-1)
    mean_vectors = pred_torch["var_5461"]
    singular_values = pred_torch["var_5465"]
    quaternions = pred_torch["var_5453"]
    colors = pred_torch["var_5456"]
    opacities = pred_torch["var_5457"]

    LOGGER.info(f"Gaussian count: {mean_vectors.shape[1]}")  # shape is (1, N, 3)

    # Create Gaussians3D from outputs
    gaussians = Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )

    # Build intrinsics
    orig_w, orig_h = original_size
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, orig_w / 2, 0],
                [0, f_px, orig_h / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_size[0] / orig_w
    intrinsics_resized[1] *= internal_size[1] / orig_h

    # Convert from NDC to metric space
    gaussians_world = unproject_gaussians(
        gaussians,
        torch.eye(4).to(device),
        intrinsics_resized,
        internal_size,
    )

    return gaussians_world


def describe_model(model: ct.models.MLModel) -> None:
    """Print model input/output descriptions.

    Args:
        model: Core ML model.
    """
    # Note: Core ML model description API varies by version
    # Just log that we're using the model
    LOGGER.info("Core ML model loaded successfully")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference on Core ML palettized model"
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Path to save output PLY file.",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        default=Path("sharp_palettized_4bit.mlpackage"),
        help="Path to Core ML model (default: sharp_palettized_4bit.mlpackage).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.model_path.exists():
        LOGGER.error(f"Model not found: {args.model_path}")
        return 1

    if not args.input_path.exists():
        LOGGER.error(f"Input image not found: {args.input_path}")
        return 1

    # Load model
    model = load_coreml_model(args.model_path)

    # Print model input/output info
    describe_model(model)

    # Load image
    LOGGER.info(f"Loading image from {args.input_path}")
    image, _, f_px = io.load_rgb(args.input_path)
    height, width = image.shape[:2]

    # Run inference
    predictions = run_inference(model, image, f_px)

    LOGGER.info("Outputs received:")
    for key, value in predictions.items():
        LOGGER.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    # Postprocess and save
    try:
        gaussians = postprocess_outputs(predictions, f_px, (width, height))
        args.output_path.parent.mkdir(exist_ok=True, parents=True)
        save_ply(gaussians, f_px, (height, width), args.output_path)
        LOGGER.info(f"Saved output to {args.output_path}")
    except Exception as e:
        LOGGER.error(f"Postprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
