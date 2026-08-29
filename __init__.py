WEB_DIRECTORY = "web"
NODE_CLASS_MAPPINGS = {}

import logging
import asyncio
import ctypes
import platform
import subprocess
import torch
import server
from aiohttp import web
import psutil
import comfy.model_management
import comfy.memory_management
import comfy.model_base


log = logging.getLogger(__name__)

try:
    import comfy_aimdo.control
except ImportError:
    comfy_aimdo = None

def _is_amd():
    # ROCm/HIP torch builds still report device.type == "cuda", so the backend
    # has to be identified through torch.version.hip.
    return getattr(torch.version, "hip", None) is not None

# NVML handle + power-cap cache. Cap and device name are static for a given
# driver state, so we only query them once. Handle init is best-effort; failures stick.
_nvml_state = {"handle": None, "tried": False, "power_limit": None}

# Vendor-agnostic; the device name never changes for the life of the process.
_gpu_name_cache = {"name": None}

def _resolve_nvml_handle(pynvml, device):
    # NVML enumerates physical GPUs and ignores CUDA_VISIBLE_DEVICES, so the torch
    # device index can point at the wrong card on multi-GPU systems. Match by UUID.
    idx = device.index if device.index is not None else torch.cuda.current_device()
    try:
        torch_uuid = "GPU-" + str(torch.cuda.get_device_properties(idx).uuid)
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(h)
            if (uuid.decode() if isinstance(uuid, bytes) else uuid) == torch_uuid:
                return h
    except Exception as e:
        log.debug("aimdo-viz: nvml uuid match failed: %s", e)
    return pynvml.nvmlDeviceGetHandleByIndex(idx)

def _nvml_handle(device):
    if _nvml_state["tried"]:
        return _nvml_state["handle"]
    _nvml_state["tried"] = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_state["handle"] = _resolve_nvml_handle(pynvml, device)
    except Exception as e:
        log.debug("aimdo-viz: pynvml init failed: %s", e)
    return _nvml_state["handle"]

def _nvml_mem_info(device):
    # On Windows WDDM, cudaMemGetInfo reports per-process memory, hiding other
    # processes' VRAM. NVML is always device-wide.
    h = _nvml_handle(device)
    if h is None:
        return None
    try:
        import pynvml
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.free, info.total
    except Exception as e:
        log.debug("aimdo-viz: nvmlDeviceGetMemoryInfo failed: %s", e)
        return None

def _nvml_power_limit(device):
    if _nvml_state["power_limit"] is not None:
        return _nvml_state["power_limit"]
    h = _nvml_handle(device)
    if h is None:
        return None
    try:
        import pynvml
        _nvml_state["power_limit"] = pynvml.nvmlDeviceGetPowerManagementLimit(h)
        return _nvml_state["power_limit"]
    except Exception as e:
        log.debug("aimdo-viz: nvmlDeviceGetPowerManagementLimit failed: %s", e)
        return None


# --- AMD telemetry -----------------------------------------------------------
# torch.cuda.utilization/temperature/power_draw go through amdsmi on a ROCm
# build, and amdsmi has no Windows port, so on Windows every hardware metric
# reads N/A. ADLX is the Windows-native equivalent, and its VRAM is device-wide
# where HIP's mem_get_info is not.
_adlx_state = {"tried": False, "ok": False, "helper": None, "perf": None,
               "gpu": None, "power_limit": None}

def _match_adlx_gpu(gpus, device):
    # ADLX ignores HIP_VISIBLE_DEVICES, so the torch index can name the wrong
    # card. UniqueId packs the PCI location as (bus << 8) | (dev << 3) | func.
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        bus = torch.cuda.get_device_properties(idx).pci_bus_id
        for g in gpus:
            if (g.UniqueId() >> 8) & 0xFF == bus:
                return g
    except Exception as e:
        log.debug("aimdo-viz: adlx pci match failed: %s", e)
    return max(gpus, key=lambda g: g.TotalVRAM())

def _adlx_init(device):
    if _adlx_state["tried"]:
        return _adlx_state["ok"]
    _adlx_state["tried"] = True
    try:
        import ADLXPybind as ADLX
        helper = ADLX.ADLXHelper()
        if helper.Initialize() != ADLX.ADLX_RESULT.ADLX_OK:
            return False
        system = helper.GetSystemServices()
        perf = system.GetPerformanceMonitoringServices() if system is not None else None
        gpus = list(system.GetGPUs()) if system is not None else []
        if perf is None or not gpus:
            return False
        # holding the helper keeps ADLX initialised
        _adlx_state.update(ok=True, helper=helper, perf=perf,
                           gpu=_match_adlx_gpu(gpus, device))
    except ImportError:
        # only actionable on Windows; elsewhere torch/amdsmi already covers it
        if platform.system() == "Windows":
            log.warning("aimdo-viz: GPU usage/temperature/power need ADLX on "
                        "Windows ROCm, install it with: pip install ADLXPybind")
        else:
            log.debug("aimdo-viz: ADLXPybind not installed")
    except Exception as e:
        log.debug("aimdo-viz: adlx init failed: %s", e)
    return _adlx_state["ok"]

def _adlx_metric(support, metrics, name, scale=1):
    # unsupported metrics still return a value, just a garbage one
    try:
        if not getattr(support, "IsSupported" + name)():
            return None
        return getattr(metrics, name)() * scale
    except Exception as e:
        log.debug("aimdo-viz: adlx %s failed: %s", name, e)
        return None

def _adlx_snapshot(device):
    """Device-wide AMD metrics, or None when ADLX is unavailable. Everything
    comes off one support/metrics pair, so it's one batched read per poll."""
    if not _adlx_init(device):
        return None
    st = _adlx_state
    try:
        support = st["perf"].GetSupportedGPUMetrics(st["gpu"])
        metrics = st["perf"].GetCurrentGPUMetrics(st["gpu"])

        mem = None
        used_mb = _adlx_metric(support, metrics, "GPUVRAM")
        if used_mb is not None:
            total = int(st["gpu"].TotalVRAM()) * 1024 * 1024
            mem = (max(0, total - int(used_mb) * 1024 * 1024), total)

        # board power covers VRAM and VRM losses; GPUPower is chip-only
        power = _adlx_metric(support, metrics, "GPUTotalBoardPower", 1000)
        rng = "GetGPUTotalBoardPowerRange"
        if power is None:
            power = _adlx_metric(support, metrics, "GPUPower", 1000)
            rng = "GetGPUPowerRange"
        if power is not None and st["power_limit"] is None:
            # range max is the cap the driver will let the board pull
            st["power_limit"] = int(getattr(support, rng)()[1]) * 1000

        util = _adlx_metric(support, metrics, "GPUUsage")
        temp = _adlx_metric(support, metrics, "GPUTemperature")
        return {
            "mem": mem,
            "util": None if util is None else round(util),
            "temp": None if temp is None else round(temp),
            "power": None if power is None else round(power),
            "power_limit": st["power_limit"],
            "name": st["gpu"].Name(),
        }
    except Exception as e:
        log.debug("aimdo-viz: adlx poll failed: %s", e)
        return None


def _get_lock():
    # Stored on comfy.model_management so the same lock survives hot reloads.
    mm = comfy.model_management
    if not hasattr(mm, '_viz_model_lock'):
        mm._viz_model_lock = asyncio.Lock()
    return mm._viz_model_lock

# cached; CPU model doesn't change at runtime.
_cpu_name_cache = {"tried": False, "name": None}

def _get_cpu_name():
    if _cpu_name_cache["tried"]:
        return _cpu_name_cache["name"]
    _cpu_name_cache["tried"] = True
    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                _cpu_name_cache["name"] = name.strip()
        elif sys_name == "Darwin":
            _cpu_name_cache["name"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2
            ).decode().strip()
        elif sys_name == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        _cpu_name_cache["name"] = line.split(":", 1)[1].strip()
                        break
    except Exception as e:
        log.debug("aimdo-viz: cpu name lookup failed: %s", e)
    return _cpu_name_cache["name"]

# Fixed drives + volume labels are static for the session — enumerate once and cache.
# Sleepy/external drives could make GetVolumeInformationW block, so do it lazily off
# the polling path (first time a client asks for the list).
_disks_cache = {"tried": False, "list": []}

def _get_volume_label(mountpoint):
    try:
        if platform.system() != "Windows":
            return None
        buf = ctypes.create_unicode_buffer(256)
        fs = ctypes.create_unicode_buffer(256)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(mountpoint), buf, 256, None, None, None, fs, 256)
        if ok:
            return buf.value or None
    except Exception as e:
        log.debug("aimdo-viz: GetVolumeInformationW failed for %s: %s", mountpoint, e)
    return None

def _get_disk_partitions():
    if _disks_cache["tried"]:
        return _disks_cache["list"]
    _disks_cache["tried"] = True
    try:
        # all=False filters CD-ROM, removable, and network on Windows
        parts = []
        for p in psutil.disk_partitions(all=False):
            parts.append({"mountpoint": p.mountpoint, "label": _get_volume_label(p.mountpoint)})
        _disks_cache["list"] = parts
    except Exception as e:
        log.debug("aimdo-viz: disk_partitions failed: %s", e)
    return _disks_cache["list"]


# Windows reads true pagefile.sys usage via NtQuerySystemInformation
# falls back to psutil on non-Windows or on any failure.
_pagefile_state = {"tried": False, "query": None}

def _build_win_pagefile_query():
    from ctypes import wintypes

    class UNICODE_STRING(ctypes.Structure):
        _fields_ = [("Length", wintypes.USHORT),
                    ("MaximumLength", wintypes.USHORT),
                    ("Buffer", wintypes.LPWSTR)]

    class SYSTEM_PAGEFILE_INFORMATION(ctypes.Structure):
        _fields_ = [("NextEntryOffset", wintypes.ULONG),
                    ("TotalSize", wintypes.ULONG),
                    ("TotalInUse", wintypes.ULONG),
                    ("PeakUsage", wintypes.ULONG),
                    ("PageFileName", UNICODE_STRING)]

    class SYSTEM_INFO(ctypes.Structure):
        _fields_ = [("wProcessorArchitecture", wintypes.WORD),
                    ("wReserved", wintypes.WORD),
                    ("dwPageSize", wintypes.DWORD),
                    ("lpMinimumApplicationAddress", ctypes.c_void_p),
                    ("lpMaximumApplicationAddress", ctypes.c_void_p),
                    ("dwActiveProcessorMask", ctypes.POINTER(wintypes.DWORD)),
                    ("dwNumberOfProcessors", wintypes.DWORD),
                    ("dwProcessorType", wintypes.DWORD),
                    ("dwAllocationGranularity", wintypes.DWORD),
                    ("wProcessorLevel", wintypes.WORD),
                    ("wProcessorRevision", wintypes.WORD)]

    SystemPageFileInformation = 18
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtQuerySystemInformation.restype = wintypes.LONG
    si = SYSTEM_INFO()
    ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(si))
    page = si.dwPageSize
    buf = ctypes.create_string_buffer(8192)
    ret = wintypes.ULONG(0)

    def query():
        status = ntdll.NtQuerySystemInformation(SystemPageFileInformation, buf, len(buf), ctypes.byref(ret))
        if status < 0 or ret.value == 0:
            return 0, 0
        total = used = 0
        off = 0
        while True:
            info = SYSTEM_PAGEFILE_INFORMATION.from_buffer(buf, off)
            total += info.TotalSize
            used += info.TotalInUse
            if info.NextEntryOffset == 0:
                break
            off += info.NextEntryOffset
        return total * page, used * page

    return query

def _pagefile_usage():
    """(total_bytes, used_bytes) of system pagefiles — true on-disk usage."""
    if platform.system() == "Windows":
        st = _pagefile_state
        if not st["tried"]:
            st["tried"] = True
            try:
                st["query"] = _build_win_pagefile_query()
            except Exception as e:
                log.debug("aimdo-viz: pagefile query init failed: %s", e)
        if st["query"] is not None:
            try:
                return st["query"]()
            except Exception as e:
                log.debug("aimdo-viz: pagefile query failed: %s", e)
    try:
        sw = psutil.swap_memory()
        return sw.total, sw.used
    except Exception:
        return 0, 0


# Hard page faults — faults that hit disk, i.e. memory paged back in. A rising
# rate is the thrashing signal. Windows reads SYSTEM_PERFORMANCE_INFORMATION;
# Linux reads /proc/vmstat pgmajfault. Returns a monotonic count; the client
# differentiates it into faults/sec.
_hardfault_state = {"tried": False, "query": None}

def _build_win_hardfault_query():
    from ctypes import wintypes

    class PERF_PREFIX(ctypes.Structure):
        _fields_ = [("IdleProcessTime", ctypes.c_longlong),
                    ("IoReadTransferCount", ctypes.c_longlong),
                    ("IoWriteTransferCount", ctypes.c_longlong),
                    ("IoOtherTransferCount", ctypes.c_longlong),
                    ("IoReadOperationCount", wintypes.ULONG),
                    ("IoWriteOperationCount", wintypes.ULONG),
                    ("IoOtherOperationCount", wintypes.ULONG),
                    ("AvailablePages", wintypes.ULONG),
                    ("CommittedPages", wintypes.ULONG),
                    ("CommitLimit", wintypes.ULONG),
                    ("PeakCommitment", wintypes.ULONG),
                    ("PageFaultCount", wintypes.ULONG),
                    ("CopyOnWriteCount", wintypes.ULONG),
                    ("TransitionCount", wintypes.ULONG),
                    ("CacheTransitionCount", wintypes.ULONG),
                    ("DemandZeroCount", wintypes.ULONG),
                    ("PageReadCount", wintypes.ULONG),
                    ("PageReadIoCount", wintypes.ULONG)]

    SystemPerformanceInformation = 2
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtQuerySystemInformation.restype = wintypes.LONG
    buf = ctypes.create_string_buffer(8192)
    ret = wintypes.ULONG(0)

    def query():
        status = ntdll.NtQuerySystemInformation(SystemPerformanceInformation, buf, len(buf), ctypes.byref(ret))
        if status < 0 or ret.value < ctypes.sizeof(PERF_PREFIX):
            return None
        return PERF_PREFIX.from_buffer(buf).PageReadIoCount

    return query

def _hard_fault_count():
    """Monotonic count of hard/major page faults, or None if unavailable."""
    sysname = platform.system()
    if sysname == "Windows":
        st = _hardfault_state
        if not st["tried"]:
            st["tried"] = True
            try:
                st["query"] = _build_win_hardfault_query()
            except Exception as e:
                log.debug("aimdo-viz: hardfault query init failed: %s", e)
        if st["query"] is not None:
            try:
                return st["query"]()
            except Exception as e:
                log.debug("aimdo-viz: hardfault query failed: %s", e)
        return None
    if sysname == "Linux":
        try:
            with open("/proc/vmstat") as f:
                for line in f:
                    if line.startswith("pgmajfault "):
                        return int(line.split()[1])
        except Exception as e:
            log.debug("aimdo-viz: /proc/vmstat read failed: %s", e)
    return None


def _vbar_residency(vbar, used_pages):
    """Per-page residency flags for the used range only. vbar.get_residency()
    builds a Python list over the whole ~10x-overallocated VBAR every poll
    (hundreds of us for large models); we run the same native fill but convert
    only the used slice. Falls back to the full read on any binding mismatch."""
    try:
        lib = comfy_aimdo.control.lib
        nr = vbar.get_nr_pages()
        buf = (ctypes.c_uint8 * nr)()
        lib.vbar_get_residency(vbar._devctx, vbar._ptr, buf, nr)
        return buf[:max(0, min(used_pages, nr))]
    except Exception as e:
        log.debug("aimdo-viz: bounded residency read failed, using full: %s", e)
        try:
            return vbar.get_residency()[:used_pages]
        except Exception:
            return []


def _detect_model_type(model_obj):
    """Classify a loaded model into a ComfyUI slot type so the UI can color it
    consistently with node connection colors. Returns None when nothing matches
    so the UI falls back to the default text color rather than mislabeling."""
    try:
        if isinstance(model_obj, comfy.model_base.BaseModel):
            return "model"
    except Exception:
        pass
    cls = model_obj.__class__
    name = cls.__name__.lower()
    module = (cls.__module__ or "").lower()
    # order matters: clip_vision before clip, style_model before model substring matches.
    if "clipvision" in name or "clip_vision" in name or "clip_vision" in module:
        return "clip_vision"
    if "controlnet" in name or "t2iadapter" in name or "controlnet" in module:
        return "controlnet"
    if "stylemodel" in name or "style_model" in name:
        return "style_model"
    if "gligen" in name:
        return "gligen"
    if "vae" in name or "autoencod" in name or module.endswith(".vae"):
        return "vae"
    if "clip" in name or "t5" in name or "textencoder" in name or "text_encoders" in module:
        return "clip"
    if "esrgan" in name or "upscal" in name or "rrdb" in name or "spandrel" in module:
        return "upscale_model"
    return None
routes = server.PromptServer.instance.routes

@routes.get("/aimdo/vram")
async def aimdo_vram_status(request):
    device = comfy.model_management.get_torch_device()
    if not torch.cuda.is_available() or device.type != "cuda":
        return web.json_response({"enabled": False})

    aimdo_active = getattr(comfy.memory_management, 'aimdo_enabled', False) and comfy_aimdo is not None

    models = []
    loaded_models = list(comfy.model_management.current_loaded_models)
    for model_idx, lm in enumerate(loaded_models):
        patcher = lm.model
        if patcher is None:
            continue

        model_obj = patcher.model
        if model_obj is None:
            continue

        name = model_obj.__class__.__name__
        is_dynamic = patcher.is_dynamic()
        total_size = patcher.model_size()
        loaded = patcher.loaded_size()

        # RAM side: pinned host memory used for fast transfers, plus
        # non-pinned loaded host memory when the patcher exposes it.
        pinned_ram = 0
        try:
            if hasattr(patcher, 'pinned_memory_size'):
                pinned_ram = patcher.pinned_memory_size()
        except Exception as e:
            log.debug("aimdo-viz: pinned_memory_size failed: %s", e)
        loaded_ram = 0
        try:
            if hasattr(patcher, 'loaded_ram_size'):
                loaded_ram = max(0, patcher.loaded_ram_size() - pinned_ram)
        except Exception as e:
            log.debug("aimdo-viz: loaded_ram_size failed: %s", e)

        # VBAR state per device (aimdo only)
        vbars = []
        vbar_loaded_total = 0

        if aimdo_active and is_dynamic and hasattr(model_obj, "dynamic_vbars"):
            for dev, vbar in model_obj.dynamic_vbars.items():
                try:
                    loaded_bytes = vbar.loaded_size()
                    vbar_loaded_total += loaded_bytes
                    page_size = getattr(vbar, 'page_size', 32 * 1024 * 1024)
                    vbar_offset = getattr(vbar, 'offset', 0)
                    if vbar_offset > 0:
                        used_pages = (vbar_offset + page_size - 1) // page_size
                    else:
                        used_pages = (total_size + page_size - 1) // page_size
                    vbars.append({
                        "device": str(dev),
                        "loaded": loaded_bytes,
                        "watermark": vbar.get_watermark(),
                        "residency": _vbar_residency(vbar, used_pages),
                    })
                except Exception as e:
                    log.warning("aimdo-viz: VBAR query failed: %s", e)

        entry = {
            "index": model_idx,
            "name": name,
            "type": _detect_model_type(model_obj),
            "total_size": total_size,
            "loaded_size": loaded,
            "vbar_loaded": vbar_loaded_total,
            "ram_size": max(0, total_size - vbar_loaded_total),
            "pinned_ram": pinned_ram,
            "loaded_ram": loaded_ram,
            "dynamic": is_dynamic,
            "vbars": vbars,
        }

        models.append(entry)

    has_dynamic = any(m.get("dynamic") for m in models)
    aimdo_usage = comfy_aimdo.control.get_total_vram_usage() if aimdo_active and has_dynamic else 0

    # Vendor telemetry: NVML on NVIDIA, ADLX on AMD. Both are device-wide, unlike
    # cudaMemGetInfo, which under-reports on Windows WDDM by hiding other
    # processes' VRAM. Without ADLX, AMD falls through to the torch calls below,
    # which work wherever amdsmi does.
    amd = _adlx_snapshot(device) if _is_amd() else None

    _mem = amd["mem"] if amd else _nvml_mem_info(device)
    if _mem is not None:
        free_cuda, total_vram = _mem
    else:
        free_cuda, total_vram = torch.cuda.mem_get_info(device)

    if amd:
        gpu_util, gpu_temp = amd["util"], amd["temp"]
        gpu_power, gpu_power_limit = amd["power"], amd["power_limit"]
    else:
        try:
            gpu_util = torch.cuda.utilization(device)
        except Exception:
            gpu_util = None

        try:
            gpu_temp = torch.cuda.temperature(device)
        except Exception:
            gpu_temp = None

        try:
            gpu_power = torch.cuda.power_draw(device)  # mW
        except Exception:
            gpu_power = None
        gpu_power_limit = _nvml_power_limit(device)  # mW

    gpu_name = _gpu_name_cache["name"]
    if gpu_name is None:
        if amd:
            gpu_name = _gpu_name_cache["name"] = amd["name"]
        else:
            h = _nvml_handle(device)
            if h is not None:
                try:
                    import pynvml
                    name = pynvml.nvmlDeviceGetName(h)
                    gpu_name = _gpu_name_cache["name"] = name.decode() if isinstance(name, bytes) else name
                except Exception as e:
                    log.debug("aimdo-viz: nvmlDeviceGetName failed: %s", e)
        if gpu_name is None:
            try:
                gpu_name = _gpu_name_cache["name"] = torch.cuda.get_device_name(device)
            except Exception:
                pass

    # non-blocking; first call after process start returns 0, subsequent calls are real
    try:
        cpu_util = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_util = None

    # pytorch internal stats
    stats = torch.cuda.memory_stats(device)
    torch_active = stats.get('active_bytes.all.current', 0)
    torch_reserved = stats.get('reserved_bytes.all.current', 0)

    ram = psutil.virtual_memory()
    proc = psutil.Process()

    process_ram = proc.memory_info().rss
    swap_total = swap_used = 0
    if request.query.get("pagefile") == "1":
        swap_total, swap_used = _pagefile_usage()

    hard_faults = None
    if request.query.get("faults") == "1":
        hard_faults = _hard_fault_count()

    disk_read = disk_write = None
    if request.query.get("disk") == "1":
        try:
            d = psutil.disk_io_counters()
            if d is not None:
                disk_read, disk_write = d.read_bytes, d.write_bytes
        except Exception:
            pass

    # client opts in by sending list_disks=1; backend skips entirely when absent.
    disks_list = None
    if request.query.get("list_disks") == "1":
        disks_list = []
        for p in _get_disk_partitions():
            entry = {"mountpoint": p["mountpoint"], "label": p["label"], "total": None, "free": None}
            try:
                du = psutil.disk_usage(p["mountpoint"])
                entry["total"] = du.total
                entry["free"] = du.free
            except Exception as e:
                log.debug("aimdo-viz: disk_usage(%s) failed: %s", p["mountpoint"], e)
            disks_list.append(entry)

    total_pinned = sum(m.get("pinned_ram", 0) for m in models)
    total_loaded_ram = sum(m.get("loaded_ram", 0) for m in models)

    return web.json_response({
        "enabled": True,
        "aimdo_active": aimdo_active,
        "total_vram": total_vram,
        "free_vram": free_cuda,
        "gpu_util": gpu_util,
        "gpu_temp": gpu_temp,
        "gpu_power": gpu_power,
        "gpu_power_limit": gpu_power_limit,
        "gpu_name": gpu_name,
        "cpu_util": cpu_util,
        "cpu_name": _get_cpu_name(),
        "aimdo_usage": aimdo_usage,
        "torch_active": torch_active,
        "torch_reserved": torch_reserved,
        "total_ram": ram.total,
        "used_ram": ram.used,
        "total_swap": swap_total,
        "used_swap": swap_used,
        "hard_faults": hard_faults,
        "disk_read": disk_read,
        "disk_write": disk_write,
        "disks": disks_list,
        "process_ram": process_ram,
        "pinned_ram": total_pinned,
        "loaded_ram": total_loaded_ram,
        "models": models,
    })

@routes.post("/aimdo/unload_all")
async def aimdo_unload_all(request):
    if _is_executing():
        return web.json_response({"error": "cannot unload during execution"}, status=409)
    async with _get_lock():
        await asyncio.get_running_loop().run_in_executor(None, comfy.model_management.unload_all_models)
    return web.json_response({"status": "ok"})


def _is_executing():
    return bool(server.PromptServer.instance.prompt_queue.currently_running)

def _get_model_idx(data):
    idx = data.get("index")
    if not isinstance(idx, int) or isinstance(idx, bool):
        return None, web.json_response({"error": "missing or invalid index"}, status=400)
    models = comfy.model_management.current_loaded_models
    if idx < 0 or idx >= len(models):
        return None, web.json_response({"error": "index out of range"}, status=400)
    return idx, None

@routes.post("/aimdo/reset_watermark")
async def aimdo_reset_watermark(request):
    idx, err = _get_model_idx(await request.json())
    if err:
        return err
    async with _get_lock():
        torch.cuda.empty_cache()
        models = comfy.model_management.current_loaded_models
        if idx >= len(models):
            return web.json_response({"error": "model no longer at index"}, status=409)
        patcher = models[idx].model
        if patcher is not None and hasattr(patcher, '_vbar_get'):
            vbar = patcher._vbar_get()
            if vbar is not None:
                vbar.prioritize()
    return web.json_response({"status": "ok"})

@routes.post("/aimdo/unload_model")
async def aimdo_unload_model(request):
    if _is_executing():
        return web.json_response({"error": "cannot unload during execution"}, status=409)
    idx, err = _get_model_idx(await request.json())
    if err:
        return err
    async with _get_lock():
        models = comfy.model_management.current_loaded_models
        if idx >= len(models):
            return web.json_response({"error": "model no longer at index"}, status=409)
        models[idx].model_unload()
        models.pop(idx)
        comfy.model_management.soft_empty_cache()
    return web.json_response({"status": "ok"})
