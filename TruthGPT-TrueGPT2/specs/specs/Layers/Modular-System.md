Advanced Modular System: Key Layers of Abstraction

| Layer               | Responsibility                                                     | Modular Feature       |
| ------------------- | ------------------------------------------------------------------ | --------------------- |
| `BaseComponent`     | Interface all layers must implement                                | Unified contract      |
| `LayerFactory`      | Dynamically register and instantiate layers (Triton or PyTorch)    | Pluggable layers      |
| `KernelRegistry`    | Register raw Triton kernels                                        | Kernel swapping       |
| `WrapperComponent`  | Adapts any kernel into a `forward()` function                      | Hardware portability  |
| `Config`            | Holds structured configs for everything (layer, kernel, optimizer) | Full parameterization |
| `ExecutionPipeline` | Controls train/eval logic                                          | Clean orchestration   |


Benefits
Swap implementations (Triton, PyTorch, custom) via config only

Central kernel management using KernelRegistry

Kernel-specific components can be benchmarked/tested in isolation

Works with any hardware backend (Triton, CUDA, CPU, etc.)

