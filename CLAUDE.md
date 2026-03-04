# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project uses `uv` for dependency management. Always use `uv run` instead of `python` when executing scripts.

```bash
# Run a script
uv run script.py

# Run the CLI
uv run sharp --help
uv run sharp predict -i data/sample.jpg -o output/

# Install dependencies
uv sync
```

## Build/Lint/Test Commands

```bash
# Linting (ruff)
uv run ruff check src/
uv run ruff check --fix src/

# Type checking (pyright)
uv run pyright

# Tests (pytest)
uv run pytest
uv run pytest -xvs tests/test_specific.py
```

## Project Overview

SHARP is a monocular view synthesis model that predicts 3D Gaussian splats from a single image in less than a second. The model uses a DINOv2-based monodepth network and a UNet-based Gaussian decoder.

### Model Architecture

The main model is `RGBGaussianPredictor` (in `src/sharp/models/predictor.py`). It consists of:

1. **Monodepth model** (`monodepth.py`): Estimates depth using DINOv2 encoders + DPT decoder
2. **Initializer** (`initializer.py`): Creates base Gaussian values from depth
3. **Feature model/Gaussian decoder** (`gaussian_decoder.py`): UNet that predicts delta values
4. **Prediction head** (`heads.py`): Decodes features to Gaussian parameters
5. **Gaussian composer** (`composer.py`): Combines base values + deltas → final Gaussians

The forward pass flow (see diagram in `predictor.py:103-192`):
```
image → monodepth → depth → init_model → base_gaussians + features
                                    ↓
features → gaussian_decoder → prediction_head → delta_values
                                    ↓
                          gaussian_composer → final_gaussians
```

### Key Files

- `src/sharp/models/predictor.py`: Main `RGBGaussianPredictor` class
- `src/sharp/models/__init__.py`: `create_predictor()` factory function
- `src/sharp/models/params.py`: `PredictorParams` dataclass for model configuration
- `src/sharp/cli/predict.py`: `sharp predict` CLI implementation
- `src/sharp/utils/gaussians.py`: `Gaussians3D` namedtuple and PLY I/O

### Coordinate System

- OpenCV convention: x right, y down, z forward
- Scene center is roughly at (0, 0, +z)
- Output PLY files follow standard 3DGS format

## Model Conversion (Core ML)

Scripts for converting PyTorch models to Core ML format:

- `convert_to_coreml_fp32.py`: Convert to FP32 Core ML model
- `palettize_model.py`: Apply 4-bit palettization then convert to Core ML
- `benchmark_all_models.py`: Benchmark PyTorch, Core ML FP32, and Core ML 4-bit

## CLI Usage

```bash
# Predict Gaussians from image(s)
uv run sharp predict -i data/sample.jpg -o output/

# Predict with rendering (requires CUDA)
uv run sharp predict -i data/sample.jpg -o output/ --render

# Use custom checkpoint
uv run sharp predict -i input/ -o output/ -c checkpoint.pt
```

The default model downloads automatically from `https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt` on first run and caches at `~/.cache/torch/hub/checkpoints/`.

## Code Style

- Python 3.13+
- Google-style docstrings
- Ruff for linting (line length 100)
- Pyright for type checking
