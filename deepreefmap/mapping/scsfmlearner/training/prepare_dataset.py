from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

from deepreefmap.camera.intrinsics import CameraProfile
from deepreefmap.camera.rectification import Rectifier


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SC-SfMLearner training dataset from MP4 videos."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more directories to recursively scan for MP4 videos.",
    )
    parser.add_argument(
        "--camera-profile",
        required=True,
        help="Camera profile name (e.g. gopro_hero_10).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=216,
        help="Target output image height in pixels.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help="Target output image width in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Target frame extraction FPS.",
    )
    return parser.parse_args()


def _find_mp4_videos(input_dirs: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for directory in input_dirs:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".mp4":
                videos.append(path)
    return sorted(videos)


def _make_unique_sequence_dir(output_root: Path, stem: str, used: set[str]) -> Path:
    candidate = stem
    counter = 1
    while candidate in used or (output_root / candidate).exists():
        candidate = f"{stem}_{counter}"
        counter += 1
    used.add(candidate)
    return output_root / candidate


def _scaled_intrinsics(profile: CameraProfile, width: int, height: int) -> np.ndarray:
    src_width, src_height = profile.image_size
    sx = float(width) / float(src_width)
    sy = float(height) / float(src_height)
    intrinsics = profile.k.astype(np.float32).copy()
    intrinsics[0, :] *= sx
    intrinsics[1, :] *= sy
    return intrinsics


def _iter_sampled_frames(video_path: Path, target_fps: int):
    meta = iio.immeta(video_path)
    src_fps = float(meta.get("fps", target_fps))
    if src_fps <= 0:
        src_fps = float(max(1, target_fps))
    stride = max(1, int(round(src_fps / max(1, target_fps))))

    for local_idx, frame in enumerate(iio.imiter(video_path)):
        if local_idx % stride == 0:
            yield frame


def _write_sequence(
    video_path: Path,
    sequence_dir: Path,
    profile: CameraProfile,
    width: int,
    height: int,
    fps: int,
    intrinsics: np.ndarray,
) -> int:
    rectifier = Rectifier(profile)
    sequence_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    for frame in _iter_sampled_frames(video_path, fps):
        rectified = rectifier.rectify(frame)
        resized = cv2.resize(rectified, (width, height), interpolation=cv2.INTER_AREA)
        image_path = sequence_dir / f"{frame_count:06d}.jpg"
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(image_path), bgr):
            raise RuntimeError(f"Failed to write image: {image_path}")
        frame_count += 1

    if frame_count == 0:
        # Keep output clean when a video cannot yield decodable frames.
        sequence_dir.rmdir()
        return 0

    np.savetxt(sequence_dir / "cam.txt", intrinsics, fmt="%.8f")
    return frame_count


def main() -> None:
    args = _parse_args()

    input_dirs = [Path(d).expanduser().resolve() for d in args.input_dirs]
    for directory in input_dirs:
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {directory}")

    if args.width <= 0 or args.height <= 0:
        raise ValueError("Width and height must be positive integers.")
    if args.fps <= 0:
        raise ValueError("FPS must be a positive integer.")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    videos = _find_mp4_videos(input_dirs)
    if not videos:
        raise RuntimeError("No MP4 files found under the provided input directories.")

    profile = CameraProfile.load(args.camera_profile)
    intrinsics = _scaled_intrinsics(profile, args.width, args.height)

    used_sequence_names: set[str] = set()
    total_frames = 0
    written_sequences = 0

    print(f"Found {len(videos)} MP4 videos.", file=sys.stderr)
    for idx, video_path in enumerate(videos, start=1):
        sequence_dir = _make_unique_sequence_dir(output_root, video_path.stem, used_sequence_names)
        print(
            f"[{idx}/{len(videos)}] Processing {video_path} -> {sequence_dir.name}",
            file=sys.stderr,
        )
        frame_count = _write_sequence(
            video_path=video_path,
            sequence_dir=sequence_dir,
            profile=profile,
            width=args.width,
            height=args.height,
            fps=args.fps,
            intrinsics=intrinsics,
        )
        if frame_count > 0:
            written_sequences += 1
            total_frames += frame_count
            print(f"    wrote {frame_count} frames", file=sys.stderr)
        else:
            print("    no frames decoded; skipped", file=sys.stderr)

    print(
        f"Done. Wrote {total_frames} frames across {written_sequences} sequences to {output_root}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
