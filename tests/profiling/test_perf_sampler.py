"""Per-stage peak memory folding and the background sampler lifecycle."""

from __future__ import annotations

from deepreefmap.profiling.perf_sampler import ResourceSample, ResourceSampler, peaks_from_marks

# Same span shape as instrumentation.py's STAGE_SPANS: (begin_mark, end_mark, stage).
_SPANS = (
    ("start", "preprocess", "startup"),
    ("preprocess", "mapping", "preprocess"),
    ("mapping", "cloud", "mapping"),
)


def test_peaks_are_the_max_within_each_stage_window() -> None:
    marks = {"start": 0.0, "preprocess": 10.0, "mapping": 20.0, "cloud": 30.0}
    samples = [
        ResourceSample(1.0, 5, 1, 0),      # startup
        ResourceSample(9.0, 7, 2, 0),      # startup (peak)
        ResourceSample(15.0, 40, 9, 1),    # preprocess (peak)
        ResourceSample(25.0, 34, 8, 6),    # mapping (swap peak, thrashing)
        ResourceSample(29.0, 30, 7, 4),    # mapping
    ]
    peaks = peaks_from_marks(samples, _SPANS, marks)
    assert peaks["startup"] == {"ram_bytes": 7, "vram_bytes": 2, "swap_bytes": 0}
    assert peaks["preprocess"] == {"ram_bytes": 40, "vram_bytes": 9, "swap_bytes": 1}
    assert peaks["mapping"] == {"ram_bytes": 34, "vram_bytes": 8, "swap_bytes": 6}


def test_stage_without_samples_or_marks_is_omitted() -> None:
    # Mapping has no marks, and preprocess has no sample landing inside it.
    marks = {"start": 0.0, "preprocess": 10.0}
    samples = [ResourceSample(1.0, 5, None)]
    peaks = peaks_from_marks(samples, _SPANS, marks)
    assert "startup" in peaks
    assert "preprocess" not in peaks and "mapping" not in peaks


def test_vram_none_when_no_sample_reported_it() -> None:
    marks = {"start": 0.0, "preprocess": 10.0}
    samples = [ResourceSample(1.0, 5, None), ResourceSample(2.0, 8, None)]
    peaks = peaks_from_marks(samples, _SPANS, marks)
    assert peaks["startup"] == {"ram_bytes": 8, "vram_bytes": None, "swap_bytes": 0}


def test_sampler_collects_samples_then_stops(monkeypatch) -> None:
    import deepreefmap.profiling.system_probe as probe

    calls = {"n": 0}

    def fake_util():
        calls["n"] += 1
        return probe.Utilisation(calls["n"], 100, 1.0, 1.0, vram_used_bytes=calls["n"], vram_total_bytes=100)

    monkeypatch.setattr(probe, "sample_utilisation", fake_util)
    sampler = ResourceSampler(interval_s=0.01)
    sampler.start()
    # Second start is a no-op (already running), must not spawn a second thread.
    sampler.start()
    import time

    time.sleep(0.1)
    sampler.stop()
    assert len(sampler.samples) >= 2
    assert sampler._thread is None
    # Samples are monotonically timestamped and carry the mocked figures.
    ts = [s.t for s in sampler.samples]
    assert ts == sorted(ts)
    assert all(s.ram_bytes >= 1 for s in sampler.samples)
