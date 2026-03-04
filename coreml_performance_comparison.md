# Core ML Model Performance Comparison

## Overview

Comparison between the 4-bit palettized model and the FP32 (full precision) original model running on Core ML.

## Test Environment

- **Device**: MacBook Air (Apple Silicon)
- **Input Image**: `data/sample.jpg`
- **Profiling Tool**: `footprint` with 0.5s sampling interval
- **Model Formats**:
  - 4-bit: `sharp_palettized_4bit.mlpackage`
  - FP32: `sharp_fp32.mlpackage`

## Performance Results

| Metric | 4-bit Palettized | FP32 Original | Improvement |
|--------|-----------------|---------------|-------------|
| **Model Load Time** | 38.05 s | 49.38 s | 23% faster |
| **Inference Time** | 3.58 s | 4.83 s | 26% faster |
| **Peak Memory** | 1,135 MB | 1,620 MB | 30% reduction |
| **Steady-State Memory** | ~370 MB | ~956 MB | 61% reduction |

## Detailed Analysis

### Load Time
The 4-bit palettized model loads **11.3 seconds faster** than the FP32 model. This is attributed to the smaller file size from 4-bit quantization, which reduces I/O overhead and model deserialization time.

### Inference Time
The 4-bit model runs inference **1.25 seconds faster** (26% improvement). This speedup comes from Core ML's ability to execute quantized operations more efficiently on the Apple Neural Engine (ANE).

### Memory Footprint

#### During Inference (Steady-State)
- **4-bit**: ~370 MB physical footprint
- **FP32**: ~956 MB physical footprint

The 4-bit model uses **61% less memory** during active inference, as the quantized weights require significantly less RAM.

#### Peak Memory
- **4-bit**: 1,135 MB peak
- **FP32**: 1,620 MB peak

The FP32 model exhibits a higher memory spike during loading and initialization, peaking at 1.62 GB compared to 1.13 GB for the quantized model.

## Conclusion

The 4-bit palettized model demonstrates significant improvements across all metrics:

1. **Faster loading** - Reduced initialization time
2. **Faster inference** - Better utilization of ANE
3. **Lower memory usage** - Critical for resource-constrained environments

For production deployments on Apple Silicon devices, the 4-bit palettized model is the recommended choice.

---

*Generated: 2026-03-04*
