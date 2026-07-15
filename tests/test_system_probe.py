"""The system probe reads memory/CPU/GPU/disk on any OS, with torch mocked."""

from __future__ import annotations

import json
import sys
import types

from deepreefmap import system_probe as sp


def _fake_torch(*, cuda=False, mps=False, name="GPU", free=1, total=2):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda,
            mem_get_info=lambda dev=0: (free, total),
            get_device_name=lambda dev=0: name,
        ),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )


def test_probe_gpu_reports_cuda_vram(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, name="RTX 4090", free=100, total=1000))
    gpu = sp._probe_gpu()
    assert gpu.kind == sp.GPU_CUDA
    assert gpu.name == "RTX 4090"
    assert (gpu.total_vram_bytes, gpu.free_vram_bytes) == (1000, 100)
    assert gpu.has_distinct_vram


def test_probe_gpu_treats_mps_as_unified(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True))
    gpu = sp._probe_gpu()
    assert gpu.kind == sp.GPU_MPS
    assert gpu.total_vram_bytes is None
    assert not gpu.has_distinct_vram


def test_probe_gpu_cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    assert sp._probe_gpu().kind == sp.GPU_NONE


def test_probe_gpu_tolerates_missing_torch(monkeypatch):
    # sys.modules[name] = None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    gpu = sp._probe_gpu()
    assert gpu.kind == sp.GPU_NONE


def test_probe_system_reports_positive_machine_figures():
    profile = sp.probe_system()
    assert profile.os_name  # "Linux" / "Windows" / "Darwin"
    assert profile.cpu_logical >= 1
    assert profile.total_ram_bytes > 0
    assert 0 <= profile.available_ram_bytes <= profile.total_ram_bytes
    assert profile.gpu.kind in (sp.GPU_CUDA, sp.GPU_MPS, sp.GPU_NONE)


def test_probe_system_bad_disk_path_is_zeroed(tmp_path):
    profile = sp.probe_system(tmp_path / "does" / "not" / "exist")
    assert profile.disk_free_bytes == 0
    assert profile.disk_total_bytes == 0


def test_profile_round_trips_through_json():
    profile = sp.probe_system()
    restored = json.loads(json.dumps(profile.to_dict()))
    assert restored["gpu"]["kind"] == profile.gpu.kind
    assert restored["total_ram_bytes"] == profile.total_ram_bytes


def test_utilisation_vram_percent():
    u = sp.Utilisation(1, 2, 50.0, 10.0, vram_used_bytes=250, vram_total_bytes=1000)
    assert u.vram_percent == 25.0
    assert sp.Utilisation(1, 2, 50.0, 10.0, None, None).vram_percent is None


def test_format_bytes():
    assert sp.format_bytes(None) == "—"
    assert sp.format_bytes(2 * 1024**3) == "2.0 GB"
    assert sp.format_bytes(512 * 1024**2) == "512 MB"
