# ComfyUI Memory Visualization

Real-time VRAM and memory monitoring panel for ComfyUI.

When [aimdo](https://github.com/Comfy-Org/comfy-aimdo/) is enabled, also shows per-model page-level residency heatmaps and watermark controls.

https://github.com/user-attachments/assets/0d212946-1309-468d-ad0f-178ea26ba257

## Installation

Clone into your `custom_nodes` folder:

```
git clone https://github.com/kijai/ComfyUI-MemoryVisualization ComfyUI/custom_nodes/ComfyUI-MemoryVisualization
```

The panel runs with no extra packages. Vendor telemetry is optional and split per vendor, so neither card's package gets installed on the other's machine:

```
# NVIDIA
pip install -r nvidia-requirements.txt

# AMD
pip install -r amd-requirements.txt
```

## GPU telemetry

Utilization, temperature and power come from the vendor telemetry library: NVML on NVIDIA and, on Linux, `amdsmi` through torch. Both are usually already present in a working ComfyUI install, and `nvidia-requirements.txt` only makes NVML explicit.

`amdsmi` has no Windows build, so AMD on Windows goes through ADLX instead. `amd-adlx` is the ADLX binding published by AMD, as prebuilt wheels for Python 3.10 to 3.13. `amd-requirements.txt` is a no-op anywhere else, since the wheels are built nowhere else and nothing else needs them.

Without the vendor package, utilization, temperature and power show `N/A`, and VRAM falls back to the torch call, which on Windows reports only this process and hides VRAM held by other applications. Everything else is unaffected.
