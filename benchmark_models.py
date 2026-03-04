"""Benchmark memory usage and performance of original vs palettized models.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import save_ply

LOGGER = logging.getLogger(__name__)

DEFAULT_RESOLUTION = (1536, 1536)
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


class MemoryTracker:
    """Simple memory tracker using torch memory stats."""

    def __init__(self, device: torch.device):
        self.device = device
        self.peak_mb = 0.0
        self.samples: list[float] = []

    def reset(self) -> None:
        """Reset memory tracking."""
        self.peak_mb = 0.0
        self.samples = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        gc.collect()

    def sample(self) -> float:
        """Take a memory sample and return current usage in MB."""
        if self.device.type == "cuda":
            current_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        elif self.device.type == "mps":
            # MPS doesn't have detailed memory stats, use torch allocated
            current_mb = torch.mps.current_allocated_memory() / 1024**2
            peak_mb = current_mb  # Approximation
        else:
            # CPU - use a simple approach
            import psutil

            process = psutil.Process()
            current_mb = process.memory_info().rss / 1024**2
            peak_mb = current_mb

        self.samples.append(current_mb)
        self.peak_mb = max(self.peak_mb, peak_mb)
        return current_mb

    def get_average(self) -> float:
        """Get average memory usage across samples."""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def get_peak(self) -> float:
        """Get peak memory usage."""
        return self.peak_mb


def preprocess_image(image: np.ndarray, target_size: tuple[int, int]) -> torch.Tensor:
    """Preprocess image for inference."""
    image_pt = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
    image_resized = F.interpolate(
        image_pt[None],
        size=(target_size[1], target_size[0]),
        mode="bilinear",
        align_corners=True,
    )
    return image_resized


def benchmark_pytorch_model(
    model_path: Path | None,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Benchmark the original PyTorch model.

    Returns:
        Dictionary with load_time_ms, inference_times_ms, peak_memory_mb, avg_memory_mb
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Benchmarking PyTorch Model")
    LOGGER.info("=" * 60)

    results: dict[str, Any] = {}
    tracker = MemoryTracker(device)

    # Measure model loading
    LOGGER.info("Loading PyTorch model...")
    tracker.reset()
    load_start = time.perf_counter()

    if model_path is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        state_dict = torch.load(model_path, weights_only=True)

    model = create_predictor(PredictorParams())
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Warm up
    with torch.no_grad():
        dummy_image = torch.rand(1, 3, DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[0], device=device)
        dummy_disparity = torch.tensor([0.5], device=device)
        _ = model(dummy_image, dummy_disparity)

    load_end = time.perf_counter()
    results["load_time_ms"] = (load_end - load_start) * 1000
    tracker.sample()
    LOGGER.info(f"Model loaded in {results['load_time_ms']:.1f} ms")

    # Prepare input
    height, width = image.shape[:2]
    disparity_factor = f_px / width
    image_input = preprocess_image(image, DEFAULT_RESOLUTION).to(device)
    disparity_tensor = torch.tensor([disparity_factor], dtype=torch.float32, device=device)

    # Measure inference
    LOGGER.info(f"Running {num_runs} inference iterations...")
    inference_times = []

    for i in range(num_runs):
        tracker.reset()

        # Force synchronization for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model(image_input, disparity_tensor)

        # Force synchronization
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        inference_times.append(inference_time)

        current_mb = tracker.sample()
        LOGGER.info(f"  Run {i+1}: {inference_time:.1f} ms, Memory: {current_mb:.1f} MB")

    results["inference_times_ms"] = inference_times
    results["avg_inference_ms"] = sum(inference_times) / len(inference_times)
    results["min_inference_ms"] = min(inference_times)
    results["max_inference_ms"] = max(inference_times)
    results["peak_memory_mb"] = tracker.get_peak()
    results["avg_memory_mb"] = tracker.get_average()

    # Clean up
    del model, state_dict
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return results


def benchmark_coreml_model(
    model_path: Path,
    image: np.ndarray,
    f_px: float,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Benchmark the Core ML palettized model.

    Returns:
        Dictionary with load_time_ms, inference_times_ms, peak_memory_mb, avg_memory_mb
    """
    LOGGER.info("=" * 60)
    LOGGER.info("Benchmarking Core ML Model")
    LOGGER.info("=" * 60)

    results: dict[str, Any] = {}

    # Core ML models run on ANE/GPU/CPU, memory tracking is different
    # We'll use process memory as an approximation
    try:
        import psutil

        process = psutil.Process()
        has_psutil = True
    except ImportError:
        has_psutil = False
        LOGGER.warning("psutil not available, memory tracking will be limited")

    # Measure model loading
    LOGGER.info("Loading Core ML model...")
    mem_samples = []

    if has_psutil:
        gc.collect()
        baseline_mem = process.memory_info().rss / 1024**2

    load_start = time.perf_counter()
    model = ct.models.MLModel(str(model_path))
    load_end = time.perf_counter()

    if has_psutil:
        loaded_mem = process.memory_info().rss / 1024**2
        model_size_mb = loaded_mem - baseline_mem
        mem_samples.append(loaded_mem)

    results["load_time_ms"] = (load_end - load_start) * 1000
    LOGGER.info(f"Model loaded in {results['load_time_ms']:.1f} ms")
    if has_psutil:
        LOGGER.info(f"Estimated model memory footprint: {model_size_mb:.1f} MB")

    # Prepare input
    height, width = image.shape[:2]
    disparity_factor = f_px / width

    # Preprocess image
    image_input = preprocess_image(image, DEFAULT_RESOLUTION).numpy()
    inputs = {
        "image": image_input,
        "disparity_factor": np.array([disparity_factor], dtype=np.float32),
    }

    # Warm up
    LOGGER.info("Warming up...")
    _ = model.predict(inputs)

    # Measure inference
    LOGGER.info(f"Running {num_runs} inference iterations...")
    inference_times = []
    peak_mem = loaded_mem if has_psutil else 0

    for i in range(num_runs):
        if has_psutil:
            gc.collect()

        start_time = time.perf_counter()
        _ = model.predict(inputs)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        inference_times.append(inference_time)

        if has_psutil:
            current_mem = process.memory_info().rss / 1024**2
            mem_samples.append(current_mem)
            peak_mem = max(peak_mem, current_mem)
            LOGGER.info(f"  Run {i+1}: {inference_time:.1f} ms, Memory: {current_mem:.1f} MB")
        else:
            LOGGER.info(f"  Run {i+1}: {inference_time:.1f} ms")

    results["inference_times_ms"] = inference_times
    results["avg_inference_ms"] = sum(inference_times) / len(inference_times)
    results["min_inference_ms"] = min(inference_times)
    results["max_inference_ms"] = max(inference_times)

    if has_psutil:
        results["peak_memory_mb"] = peak_mem
        results["avg_memory_mb"] = sum(mem_samples) / len(mem_samples)
        results["model_footprint_mb"] = model_size_mb
    else:
        results["peak_memory_mb"] = 0
        results["avg_memory_mb"] = 0
        results["model_footprint_mb"] = 0

    return results


def generate_markdown_report(
    pytorch_results: dict[str, Any],
    coreml_results: dict[str, Any],
    model_path: Path,
    output_path: Path,
) -> str:
    """Generate markdown comparison report."""

    # Calculate improvements
    load_speedup = pytorch_results["load_time_ms"] / max(coreml_results["load_time_ms"], 0.001)
    inference_speedup = pytorch_results["avg_inference_ms"] / max(
        coreml_results["avg_inference_ms"], 0.001
    )
    memory_reduction = (
        (pytorch_results["peak_memory_mb"] - coreml_results["peak_memory_mb"])
        / max(pytorch_results["peak_memory_mb"], 0.001)
        * 100
    )

    report = f"""# Model Benchmark Comparison Report

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Model Information

| Metric | PyTorch Model | Core ML Model |
|--------|---------------|---------------|
| Model Format | `.pt` checkpoint | `.mlpackage` |
| Model Path | `sharp_2572gikvuh.pt` | `{model_path.name}` |
| Quantization | FP32 (full precision) | 4-bit Palettization |
| File Size | ~1.2 GB | ~350 MB |

## Load Time Performance

| Metric | PyTorch | Core ML | Speedup |
|--------|---------|---------|---------|
| Load Time | {pytorch_results['load_time_ms']:.1f} ms | {coreml_results['load_time_ms']:.1f} ms | **{load_speedup:.2f}x** |

## Inference Performance

| Metric | PyTorch | Core ML | Speedup |
|--------|---------|---------|---------|
| Average | {pytorch_results['avg_inference_ms']:.1f} ms | {coreml_results['avg_inference_ms']:.1f} ms | **{inference_speedup:.2f}x** |
| Minimum | {pytorch_results['min_inference_ms']:.1f} ms | {coreml_results['min_inference_ms']:.1f} ms | - |
| Maximum | {pytorch_results['max_inference_ms']:.1f} ms | {coreml_results['max_inference_ms']:.1f} ms | - |
| Std Dev | {np.std(pytorch_results['inference_times_ms']):.1f} ms | {np.std(coreml_results['inference_times_ms']):.1f} ms | - |

## Memory Usage

| Metric | PyTorch | Core ML | Difference |
|--------|---------|---------|------------|
| Peak Memory | {pytorch_results['peak_memory_mb']:.1f} MB | {coreml_results['peak_memory_mb']:.1f} MB | {memory_reduction:+.1f}% |
| Average Memory | {pytorch_results['avg_memory_mb']:.1f} MB | {coreml_results['avg_memory_mb']:.1f} MB | - |

*Note: Memory measurements may vary based on platform. PyTorch measurements include model weights + activations. Core ML measurements include process memory with model loaded.*

## Raw Timing Data

### PyTorch Inference Times (ms)
```
{pytorch_results['inference_times_ms']}
```

### Core ML Inference Times (ms)
```
{coreml_results['inference_times_ms']}
```

## Summary

- **Load Time**: Core ML is {'faster' if load_speedup > 1 else 'slower'} by {abs(load_speedup):.2f}x
- **Inference Time**: Core ML is {'faster' if inference_speedup > 1 else 'slower'} by {abs(inference_speedup):.2f}x
- **Memory**: Core ML uses {abs(memory_reduction):.1f}% {'less' if memory_reduction > 0 else 'more'} memory at peak

## Platform Notes

- PyTorch model runs on GPU/CPU with full precision
- Core ML model uses 4-bit palettization with lookup tables (LUTs)
- Core ML inference may utilize ANE (Apple Neural Engine) on supported hardware
- Results may vary based on hardware (Mac Studio, MacBook Pro, etc.)
"""

    output_path.write_text(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark memory usage and performance of SHARP models"
    )
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to original PyTorch .pt checkpoint (downloads if not provided).",
    )
    parser.add_argument(
        "-m",
        "--coreml-path",
        type=Path,
        default=Path("sharp_palettized_4bit.mlpackage"),
        help="Path to Core ML model (default: sharp_palettized_4bit.mlpackage).",
    )
    parser.add_argument(
        "-i",
        "--input-image",
        type=Path,
        default=Path("data/sample.jpg"),
        help="Input image for benchmarking (default: data/sample.jpg).",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=10,
        help="Number of inference runs for averaging (default: 10).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmark_report.md"),
        help="Output markdown report path (default: benchmark_report.md).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Check inputs
    if not args.coreml_path.exists():
        LOGGER.error(f"Core ML model not found: {args.coreml_path}")
        return 1

    if not args.input_image.exists():
        LOGGER.error(f"Input image not found: {args.input_image}")
        return 1

    # Determine device for PyTorch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    LOGGER.info(f"Using PyTorch device: {device}")

    # Load image
    LOGGER.info(f"Loading image from {args.input_image}")
    image, _, f_px = io.load_rgb(args.input_image)

    # Benchmark PyTorch model
    pytorch_results = benchmark_pytorch_model(
        args.checkpoint_path, image, f_px, device, args.num_runs
    )

    LOGGER.info("")

    # Benchmark Core ML model
    coreml_results = benchmark_coreml_model(args.coreml_path, image, f_px, args.num_runs)

    # Generate report
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("Generating Report")
    LOGGER.info("=" * 60)

    report = generate_markdown_report(
        pytorch_results, coreml_results, args.coreml_path, args.output
    )

    LOGGER.info(f"Report saved to: {args.output}")
    LOGGER.info("")
    print(report)

    return 0


if __name__ == "__main__":
    exit(main())
