from pathlib import Path

from deepreefmap.io.video_hash import hash_video, hash_videos


def test_hash_is_stable_across_calls(tmp_path: Path) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x01\x02" * 200_000)
    assert hash_video(f) == hash_video(f)


def test_hash_is_content_based(tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"\x01" * 300_000)
    b.write_bytes(a.read_bytes())
    assert hash_video(a) == hash_video(b)

    b.write_bytes(b"\x02" * 300_000)
    assert hash_video(a) != hash_video(b)


def test_hash_survives_rename(tmp_path: Path) -> None:
    f = tmp_path / "original.mp4"
    f.write_bytes(b"\x03" * 300_000)
    before = hash_video(f)
    renamed = f.rename(tmp_path / "renamed.mp4")
    assert hash_video(renamed) == before


def test_hash_failure_returns_none(tmp_path: Path) -> None:
    assert hash_video(tmp_path / "missing.mp4") is None


def test_hash_videos_is_parallel_to_input(tmp_path: Path) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x04" * 300_000)
    hashes = hash_videos([f, tmp_path / "missing.mp4"])
    assert len(hashes) == 2
    assert isinstance(hashes[0], str) and len(hashes[0]) == 32
    assert hashes[1] is None
