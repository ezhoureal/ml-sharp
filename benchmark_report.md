# Model Benchmark Comparison Report

Generated: 2026-03-03 20:54:13

## Model Information

| Metric | PyTorch Model | Core ML 4.0bpw Model |
|--------|---------------|---------------|
| Model Format | `.pt` checkpoint | `.mlpackage` |
| Model Path | `sharp_2572gikvuh.pt` | `sharp_palettized_4bit.mlpackage` |
| Quantization | FP32 (full precision) | 4-bit Palettization |
| File Size | ~1.2 GB | ~350 MB |

## Load Time Performance

| Metric | PyTorch | Core ML 4.0bpw | Speedup |
|--------|---------|---------|---------|
| Load Time | 19996.8 ms | 42479.2 ms | **0.47x** |

## Inference Performance

| Metric | PyTorch | Core ML 4.0bpw | Speedup |
|--------|---------|---------|---------|
| Average | 60630.2 ms | 3329.1 ms | **18.21x** |
| Minimum | 15398.7 ms | 3216.6 ms | - |
| Maximum | 90862.3 ms | 3462.5 ms | - |
| Std Dev | 28970.4 ms | 94.8 ms | - |

## Memory Usage

| Metric | PyTorch | Core ML 4.0bpw | Difference |
|--------|---------|---------|------------|
| Peak Memory | 2935.0 MB | 1247.7 MB | +57.5% |
| Average Memory | 2935.0 MB | 1073.3 MB | - |

*Note: Memory measurements may vary based on platform. PyTorch measurements include model weights + activations. Core ML 4.0bpw measurements include process memory with model loaded.*

## Raw Timing Data

### PyTorch Inference Times (ms)
```
[15398.69883400388, 38336.09316701768, 83974.91354105296, 74578.76358402427, 90862.28962504538]
```

### Core ML 4.0bpw Inference Times (ms)
```
[3371.718833979685, 3216.5571249788627, 3224.333541991655, 3462.540959008038, 3370.1167079852894]
```

## Summary

- **Load Time**: Core ML 4.0bpw is slower by 0.47x
- **Inference Time**: Core ML 4.0bpw is faster by 18.21x
- **Memory**: Core ML 4.0bpw uses 57.5% less memory at peak

## Platform Notes

- PyTorch model runs on GPU/CPU with full precision
- Core ML 4.0bpw model uses 4-bit palettization with lookup tables (LUTs)
- Core ML 4.0bpw inference may utilize ANE (Apple Neural Engine) on supported hardware
- Results may vary based on hardware (Mac Studio, MacBook Pro, etc.)
