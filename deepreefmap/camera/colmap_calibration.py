from pathlib import Path
import tempfile
import shutil

import cv2
import imageio.v3 as iio
import numpy as np

from deepreefmap.camera.intrinsics import CameraProfile
from deepreefmap.camera.rectification import Rectifier


def _camera_model_name(camera: object) -> str:
    """Best-effort camera model extraction across pycolmap versions."""
    for attr in ("model_name", "model"):
        val = getattr(camera, attr, None)
        if val is None:
            continue
        # model can be an enum, object with name, or direct string.
        name = getattr(val, "name", val)
        text = str(name).upper().strip()
        if not text:
            continue
        # Some enums stringify as "CameraModelId.RADIAL".
        if "." in text:
            text = text.split(".")[-1]
        return text
    return "UNKNOWN"


def _camera_model_debug(camera: object) -> dict[str, object]:
    """Collect camera fields useful for diagnosing pycolmap API differences."""
    debug: dict[str, object] = {"camera_type": type(camera).__name__}
    for key in ("model_name", "model", "camera_id", "params"):
        if hasattr(camera, key):
            try:
                value = getattr(camera, key)
                if key == "params":
                    value = list(value)
                debug[key] = value
            except Exception:
                debug[key] = "<unreadable>"
    return debug


def _parse_colmap_camera(camera: object) -> tuple[str, dict[str, float], np.ndarray, np.ndarray]:
    """Parse supported COLMAP camera models into a common representation."""
    model_name = _camera_model_name(camera)
    params = [float(v) for v in getattr(camera, "params", [])]
    if not params:
        raise RuntimeError("COLMAP returned camera without parameters.")

    if model_name in {"RADIAL", "SIMPLE_RADIAL"}:
        # SIMPLE_RADIAL: [f, cx, cy, k]
        # RADIAL: [f, cx, cy, k1, k2] (sometimes expanded to fx/fy forms)
        if model_name == "SIMPLE_RADIAL" and len(params) >= 4:
            f, cx, cy, k1 = params[:4]
            fx, fy, k2 = f, f, 0.0
        elif len(params) == 5:
            f, cx, cy, k1, k2 = params
            fx, fy = f, f
        elif len(params) >= 6:
            fx, fy, cx, cy, k1, k2 = params[:6]
        else:
            raise RuntimeError(f"Unexpected {model_name} camera parameters: {params}")
        distorted = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2}
        k_dist = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
        return model_name, distorted, k_dist, dist

    if model_name == "OPENCV_FISHEYE":
        # OPENCV_FISHEYE: [fx, fy, cx, cy, k1, k2, k3, k4]
        if len(params) < 8:
            raise RuntimeError(f"Unexpected OPENCV_FISHEYE camera parameters: {params}")
        fx, fy, cx, cy, k1, k2, k3, k4 = params[:8]
        distorted = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "k3": k3, "k4": k4}
        k_dist = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist = np.array([k1, k2, k3, k4], dtype=np.float32)
        return model_name, distorted, k_dist, dist

    if model_name in {"PINHOLE", "SIMPLE_PINHOLE"}:
        if model_name == "SIMPLE_PINHOLE" and len(params) >= 3:
            f, cx, cy = params[:3]
            fx, fy = f, f
        elif len(params) >= 4:
            fx, fy, cx, cy = params[:4]
        else:
            raise RuntimeError(f"Unexpected {model_name} camera parameters: {params}")
        distorted = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        k_dist = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist = np.zeros(4, dtype=np.float32)
        return model_name, distorted, k_dist, dist

    raise RuntimeError(
        f"Unsupported COLMAP camera model {model_name}. "
        "Supported models: RADIAL, SIMPLE_RADIAL, OPENCV_FISHEYE, PINHOLE, SIMPLE_PINHOLE."
    )


def _sample_video_frames(
    video_path: Path,
    out_dir: Path,
    n_frames: int,
    fps: int,
    begin_s: float | None = None,
    end_s: float | None = None,
) -> list[Path]:
    meta = iio.immeta(video_path)
    src_fps = float(meta.get("fps", fps))
    duration = float(meta.get("duration", 0.0) or 0.0)
    start_t = max(0.0, begin_s if begin_s is not None else 0.0)
    end_t = end_s if end_s is not None else duration
    if duration > 0.0:
        end_t = min(end_t, duration)
    if end_t <= start_t:
        raise RuntimeError(f"Invalid calibration timestamp range: begin={start_t}, end={end_t}")

    start_idx = int(round(start_t * src_fps))
    end_idx = int(round(end_t * src_fps)) if duration > 0.0 or end_s is not None else None
    stride = max(1, int(round(src_fps / max(1, fps))))
    selected: list[np.ndarray] = []
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx < start_idx:
            continue
        if end_idx is not None and idx > end_idx:
            break
        if idx % stride != 0:
            continue
        selected.append(frame)
        if len(selected) >= n_frames:
            break
    if not selected:
        raise RuntimeError("No frames found in video at requested sampling rate.")
    out_paths: list[Path] = []
    for i, frame in enumerate(selected):
        p = out_dir / f"{i:06d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_paths.append(p)
    return out_paths


def calibrate_camera_profile(
    video: Path,
    name: str,
    n_frames: int = 100,
    fps: int = 10,
    begin_s: float | None = None,
    end_s: float | None = None,
) -> Path:
    with tempfile.TemporaryDirectory(prefix="drm_calib_") as tmp:
        tmp_dir = Path(tmp)
        image_dir = tmp_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        sample_paths = _sample_video_frames(
            video,
            image_dir,
            n_frames=n_frames,
            fps=fps,
            begin_s=begin_s,
            end_s=end_s,
        )
        h, w = cv2.imread(str(sample_paths[0])).shape[:2]

        try:
            import pycolmap
        except Exception as exc:
            raise RuntimeError("Calibration requires pycolmap to be installed and importable.") from exc

        database_path = tmp_dir / "database.db"
        sparse_path = tmp_dir / "sparse"
        sparse_path.mkdir(parents=True, exist_ok=True)

        reader_options = pycolmap.ImageReaderOptions()
        if hasattr(reader_options, "camera_model"):
            reader_options.camera_model = "RADIAL"

        # pycolmap API differs by version. Newer versions support camera_mode,
        # while some expose single_camera on reader options.
        extract_kwargs = {
            "database_path": str(database_path),
            "image_path": str(image_dir),
            "reader_options": reader_options,
        }
        if hasattr(pycolmap, "CameraMode"):
            extract_kwargs["camera_mode"] = pycolmap.CameraMode.SINGLE
        elif hasattr(reader_options, "single_camera"):
            reader_options.single_camera = True
        pycolmap.extract_features(**extract_kwargs)
        pycolmap.match_sequential(database_path=str(database_path))
        maps = pycolmap.incremental_mapping(database_path=str(database_path), image_path=str(image_dir), output_path=str(sparse_path))
        if not maps:
            raise RuntimeError("COLMAP mapping failed: no reconstruction produced.")

        # Pick largest reconstruction by registered images.
        best_rec = max(maps.values(), key=lambda rec: len(rec.images))
        if len(best_rec.images) < max(10, len(sample_paths) // 3):
            raise RuntimeError(
                f"Calibration failed quality gate: only {len(best_rec.images)} registered images out of {len(sample_paths)}."
            )
        cam = next(iter(best_rec.cameras.values()))
        try:
            model_name, distorted, k_dist, dist = _parse_colmap_camera(cam)
        except Exception as exc:
            debug_fields = _camera_model_debug(cam)
            raise RuntimeError(f"Failed to parse COLMAP camera: {debug_fields}") from exc

        # Prefer RADIAL at extraction time, but accept compatible fallback models.
        if model_name in {"RADIAL", "SIMPLE_RADIAL"}:
            k_rect, roi = cv2.getOptimalNewCameraMatrix(
                k_dist,
                dist,
                (w, h),
                alpha=0.0,
                newImgSize=(w, h),
                centerPrincipalPoint=True,
            )
            k_rect = k_rect.astype(np.float32)
        elif model_name == "OPENCV_FISHEYE":
            k_rect = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                k_dist,
                dist.reshape(4, 1),
                (w, h),
                R=np.eye(3, dtype=np.float32),
                balance=0.0,
                new_size=(w, h),
            ).astype(np.float32)
            roi = (0, 0, w, h)
        else:
            # PINHOLE and SIMPLE_PINHOLE are already undistorted.
            k_rect = k_dist.astype(np.float32)
            roi = (0, 0, w, h)

        mean_reproj = None
        if hasattr(best_rec, "compute_mean_reprojection_error"):
            try:
                mean_reproj = float(best_rec.compute_mean_reprojection_error())
            except Exception:
                mean_reproj = None

        diagnostics = {
            "n_input_frames": len(sample_paths),
            "n_registered_images": len(best_rec.images),
            "mean_reprojection_error_px": mean_reproj,
            "camera_model": model_name,
            "rectification_mode": "none" if model_name in {"PINHOLE", "SIMPLE_PINHOLE"} else "undistort",
            "source_video": video.name,
            "sampling_fps": fps,
            "begin_s": begin_s,
            "end_s": end_s,
            "valid_roi_xywh": [int(v) for v in roi],
        }
        profile = CameraProfile(
            name=name,
            image_size=(w, h),
            k=k_rect,
            distorted_model=model_name,
            radial=distorted,
            diagnostics=diagnostics,
        )
        profile_path = profile.save()
        rectifier = Rectifier(profile)

        diagnostics_dir = Path("camera_profiles") / f"{name}_diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        sparse_diag_dir = diagnostics_dir / "sparse"
        sparse_diag_dir.mkdir(parents=True, exist_ok=True)
        try:
            best_rec.write(str(sparse_diag_dir))
        except Exception:
            pass
        for p in sample_paths[:4]:
            shutil.copy2(p, diagnostics_dir / f"raw_{p.name}")
            raw_bgr = cv2.imread(str(p))
            if raw_bgr is None:
                continue
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            rect_rgb = rectifier.rectify(raw_rgb)
            side_by_side = np.concatenate([raw_rgb, rect_rgb], axis=1)
            cv2.imwrite(str(diagnostics_dir / f"compare_{p.name}"), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        return profile_path


def verify_camera_profile(name: str) -> dict[str, object]:
    profile = CameraProfile.load(name)
    diagnostics_dir = Path("camera_profiles") / f"{name}_diagnostics"
    previews = sorted(str(p) for p in diagnostics_dir.glob("*.png")) if diagnostics_dir.exists() else []
    return {
        "profile": name,
        "image_size": profile.image_size,
        "k": profile.k.tolist(),
        "radial": profile.radial,
        "diagnostics": profile.diagnostics,
        "diagnostic_previews": previews[:4],
        "diagnostics_dir_exists": diagnostics_dir.exists(),
    }
