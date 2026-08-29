# ComfyUI Memory Visualization

Real-time VRAM and memory monitoring panel for ComfyUI.

When [aimdo](https://github.com/Comfy-Org/comfy-aimdo/) is enabled, also shows per-model page-level residency heatmaps and watermark controls.

https://github.com/user-attachments/assets/0d212946-1309-468d-ad0f-178ea26ba257

## Installation

Clone into your `custom_nodes` folder:

```
git clone https://github.com/kijai/ComfyUI-MemoryVisualization ComfyUI/custom_nodes/ComfyUI-MemoryVisualization
```

## GPU telemetry

Utilization, temperature and power come from the vendor telemetry library: NVML on NVIDIA and, on Linux, `amdsmi` through torch - both already present in a working ComfyUI install.

`amdsmi` has no Windows build, so AMD on Windows needs ADLX instead:

```
pip install ADLXPybind
```

Without it those bars show `N/A`.
