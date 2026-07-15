"""Cross-platform machine probe: RAM, VRAM, CPU and disk, with no video needed.

Field laptops crash hard when a run exhausts RAM: on Linux the kernel OOM killer
sends an uncatchable SIGKILL, so the only real control is to measure the machine
up front and refuse a run that will not fit. This module is the measurement
layer. It works on Linux, Windows and macOS and in the packaged binaries, and it
is Qt-free so it can be probed headless and unit-tested by mocking psutil/torch.

VRAM is device-specific. A CUDA (or ROCm-shimmed) GPU reports a real free/total
pair. Apple's MPS shares one unified pool with system RAM, so there is no
separate VRAM figure and callers should budget against system RAM instead.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

# GPU memory kinds. "cuda" covers NVIDIA and the ROCm shim (both expose the
# torch.cuda API); "mps" is Apple unified memory (no distinct VRAM pool);
# "none" is CPU-only.
GPU_CUDA = "cuda"
GPU_MPS = "mps"
GPU_NONE = "none"


@dataclass(frozen=True)
class GpuInfo:
    kind: str
    name: str
    total_vram_bytes: int | None  # None when unified (MPS) or unknown
    free_vram_bytes: int | None

    @property
    def has_distinct_vram(self) -> bool:
        """True when the GPU has its own memory pool to budget separately from RAM."""
        return self.kind == GPU_CUDA and self.total_vram_bytes is not None


@dataclass(frozen=True)
class SystemProfile:
    os_name: str
    os_release: str
    cpu_logical: int
    cpu_physical: int | None
    total_ram_bytes: int
    available_ram_bytes: int
    total_swap_bytes: int
    free_swap_bytes: int
    gpu: GpuInfo
    disk_total_bytes: int
    disk_free_bytes: int
    disk_path: str

    def to_dict(self) -> dict:
        """Flatten to a JSON-serialisable dict for run_timings metadata."""
        return asdict(self)


def _probe_gpu() -> GpuInfo:
    """Read GPU kind, name and (where it exists) VRAM, tolerating any failure.

    torch is imported lazily so the RAM/CPU/disk probe stays usable when torch is
    absent or slow to load, and so a missing GPU never turns into a hard error on
    a field laptop.
    """
    try:
        import torch
    except Exception:
        return GpuInfo(GPU_NONE, "CPU only (torch unavailable)", None, None)

    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return GpuInfo(GPU_CUDA, torch.cuda.get_device_name(0), int(total), int(free))
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            # Apple Silicon: the GPU draws from system RAM, so there is no
            # separate VRAM figure to report.
            return GpuInfo(GPU_MPS, "Apple GPU (unified memory)", None, None)
    except Exception:
        pass
    return GpuInfo(GPU_NONE, "CPU only", None, None)


def probe_system(disk_path: Path | str | None = None) -> SystemProfile:
    """Snapshot the machine's memory, CPU, GPU and free disk on the output volume.

    ``disk_path`` selects which volume's free space is reported (the run output
    directory in practice); it defaults to the current working directory.
    """
    vm = psutil.virtual_memory()
    try:
        sw = psutil.swap_memory()
        swap_total, swap_free = int(sw.total), int(sw.free)
    except Exception:
        swap_total = swap_free = 0
    path = Path(disk_path) if disk_path is not None else Path.cwd()
    try:
        du = psutil.disk_usage(str(path))
        disk_total, disk_free = int(du.total), int(du.free)
    except OSError:
        disk_total = disk_free = 0
    return SystemProfile(
        os_name=platform.system(),
        os_release=platform.release(),
        cpu_logical=psutil.cpu_count(logical=True) or 1,
        cpu_physical=psutil.cpu_count(logical=False),
        total_ram_bytes=int(vm.total),
        available_ram_bytes=int(vm.available),
        total_swap_bytes=swap_total,
        free_swap_bytes=swap_free,
        gpu=_probe_gpu(),
        disk_total_bytes=disk_total,
        disk_free_bytes=disk_free,
        disk_path=str(path),
    )


@dataclass(frozen=True)
class Utilisation:
    ram_used_bytes: int
    ram_total_bytes: int
    ram_percent: float
    cpu_percent: float
    vram_used_bytes: int | None
    vram_total_bytes: int | None

    @property
    def vram_percent(self) -> float | None:
        if self.vram_used_bytes is None or not self.vram_total_bytes:
            return None
        return 100.0 * self.vram_used_bytes / self.vram_total_bytes


def sample_utilisation() -> Utilisation:
    """A cheap live snapshot for the GUI gauges, safe to call on a timer.

    ``cpu_percent`` is the non-blocking reading (percent since the previous call),
    so drive it from a steady interval rather than a one-off call for a real
    figure.
    """
    vm = psutil.virtual_memory()
    vram_used = vram_total = None
    gpu = _probe_gpu()
    if gpu.has_distinct_vram and gpu.total_vram_bytes is not None and gpu.free_vram_bytes is not None:
        vram_total = gpu.total_vram_bytes
        vram_used = gpu.total_vram_bytes - gpu.free_vram_bytes
    return Utilisation(
        ram_used_bytes=int(vm.total - vm.available),
        ram_total_bytes=int(vm.total),
        ram_percent=float(vm.percent),
        cpu_percent=psutil.cpu_percent(interval=None),
        vram_used_bytes=vram_used,
        vram_total_bytes=vram_total,
    )


def format_bytes(n: int | None) -> str:
    """Render a byte count as `3.4 GB` / `812 MB`, or `—` for unknown."""
    if n is None:
        return "—"
    gb = n / 1024**3
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{n / 1024**2:.0f} MB"
