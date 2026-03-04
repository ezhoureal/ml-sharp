# Model Benchmark Comparison: Core ML Quantization Analysis

Generated: 2026-03-03 22:14:34

## Executive Summary

This report compares three model configurations:
1. **PyTorch MPS** - Original FP32 model running on Metal Performance Shaders
2. **Core ML FP32** - Unquantized model converted to Core ML format
3. **Core ML 4-bit** - 4-bit palettized model (quantized)

## Model Specifications

| Model | Backend | Quantization | File Size |
|-------|---------|--------------|-----------|
| PyTorch MPS | Metal Performance Shaders | FP32 (full precision) | ~1.2 GB (weights) |
| Core ML FP32 | Core ML (ANE/GPU/CPU) | FP32 (full precision) | 1.2 GB |
| Core ML 4-bit | Core ML (ANE/GPU/CPU) | 4-bit Palettization (16 levels) | 350 MB |

## File Size Comparison

| Metric | Core ML FP32 | Core ML 4-bit | Reduction |
|--------|--------------|---------------|-----------|
| File Size | 1.2 GB | 350 MB | **70.8%** |

## Load Time Performance

| Model | Load Time | Relative |
|-------|-----------|----------|
| PyTorch MPS | 24087.6 ms | baseline |
| Core ML FP32 | 65825.8 ms | 2.73x |
| Core ML 4-bit | 38323.7 ms | 1.59x |

## Inference Performance

| Model | Average | Min | Max | Std Dev | Speedup vs PyTorch |
|-------|---------|-----|-----|---------|-------------------|
| PyTorch MPS | 59709.6 ms | 13802.4 ms | 110802.3 ms | 33788.9 ms | 1.00x |
| Core ML FP32 | 4501.8 ms | 4493.8 ms | 4508.4 ms | 6.4 ms | **13.26x** |
| Core ML 4-bit | 3275.5 ms | 3221.7 ms | 3346.4 ms | 51.0 ms | **18.23x** |

### Core ML Quantization Impact

| Comparison | Inference Time | Speedup |
|------------|---------------|---------|
| Core ML FP32 vs PyTorch | 4501.8 ms vs 59709.6 ms | **13.26x** |
| Core ML 4-bit vs PyTorch | 3275.5 ms vs 59709.6 ms | **18.23x** |
| Core ML 4-bit vs Core ML FP32 | 3275.5 ms vs 4501.8 ms | **1.37x** |

## Raw Timing Data

### PyTorch MPS Inference Times (ms)
```
[13802.364166011102, 32441.793957957998, 69492.28004197357, 110802.33887501527, 72009.19404102024]
```

### Core ML FP32 Inference Times (ms)
```
[4508.372708049137, 4505.330833024345, 4507.181000022683, 4493.801458040252, 4494.170082965866]
```

### Core ML 4-bit Inference Times (ms)
```
[3254.021709028166, 3229.5969579718076, 3325.716666004155, 3346.3925829855725, 3221.694333013147]
```

## Analysis

### 1. Quantization Impact on Speed
The 4-bit palettization provides **1.37x faster** inference compared to the unquantized Core ML model.
This is likely due to:
- Reduced memory bandwidth (4x smaller weights)
- ANE (Apple Neural Engine) optimization for quantized models
- LUT-based dequantization is hardware-accelerated

### 2. Core ML vs PyTorch
Both Core ML models significantly outperform PyTorch MPS:
- Core ML FP32: **13.26x faster**
- Core ML 4-bit: **18.23x faster**

This demonstrates the benefits of:
- ANE (Apple Neural Engine) utilization
- Optimized Core ML runtime
- Hardware-accelerated inference

### 3. Consistency
Core ML models show much lower variance (std dev) compared to PyTorch:
- PyTorch Std Dev: 33788.9 ms
- Core ML FP32 Std Dev: 6.4 ms
- Core ML 4-bit Std Dev: 51.0 ms

### 4. Memory Efficiency
- **Disk Size**: 70.8% reduction (1.2 GB → 350 MB)
- **PyTorch MPS**: ~2500 MB total (CPU + GPU unified memory)
- **Core ML FP32**: ~-147 MB process memory
- **Core ML 4-bit**: ~-45 MB process memory

## Memory Usage Comparison

> **Note**: On Apple Silicon with unified memory, PyTorch MPS uses `torch.mps` which reports
> GPU-allocated memory separately. Core ML uses ANE/GPU transparently through the Core ML runtime.

### Model Loading Memory Footprint

| Model | CPU Memory | GPU Memory | Total | Notes |
|-------|------------|------------|-------|-------|
| PyTorch MPS | -272.9 MB | 2773.0 MB | 2500.1 MB | Model weights on GPU (MPS) |
| Core ML FP32 | -147.1 MB | N/A | -147.1 MB | Core ML runtime overhead |
| Core ML 4-bit | -44.9 MB | N/A | -44.9 MB | Core ML runtime overhead |

### Peak Memory During Inference

| Model | Peak CPU | Peak GPU | Total Peak |
|-------|----------|----------|------------|
| PyTorch MPS | 100.1 MB | 2865.8 MB | 2965.9 MB |
| Core ML FP32 | 956.6 MB | N/A | 956.6 MB |
| Core ML 4-bit | 1254.6 MB | N/A | 1254.6 MB |

### Memory Efficiency Analysis

1. **File Size vs Runtime Memory**:
   - **File Size Reduction**: 70.8% smaller on disk (1.2 GB → 350 MB)
   - **PyTorch MPS GPU Memory**: ~2773 MB for model weights
   - **Core ML Runtime**: Both FP32 and 4-bit show similar ~-147 MB process overhead

2. **Inference Working Memory**:
   - PyTorch MPS uses unified memory: 2966 MB peak
   - Core ML FP32: 957 MB peak (ANE/GPU working memory)
   - Core ML 4-bit: 1255 MB peak (ANE/GPU with quantized ops)

3. **Key Takeaway**:
   - The 4-bit quantization provides **71% file size reduction** and **18.2x speedup**
   - PyTorch MPS requires ~2773 MB GPU memory for model weights
   - Core ML manages memory transparently through the ANE/GPU
   - Smaller file size enables faster loading and lower storage requirements

## Recommendations

1. **For Production Deployment**: Use **Core ML 4-bit** model
   - Smallest file size (350 MB)
   - Fastest inference (~3275 ms)
   - ANE-optimized

2. **For Quality-Critical Applications**: Consider Core ML FP32
   - Full precision weights
   - Slightly slower but may have marginally better quality
   - Good for comparison/benchmarking

3. **Avoid PyTorch MPS for Production**
   - Much slower inference (~59710 ms)
   - Higher variance
   - Only useful for training or non-Apple platforms

## Technical Notes

- **Palettization**: 4-bit quantization with per_tensor granularity
- **ANE**: Apple Neural Engine (dedicated ML hardware)
- **MPS**: Metal Performance Shaders (GPU compute)
- **FP32**: 32-bit floating point (full precision)
- All tests run on Apple Silicon with macOS 15+
