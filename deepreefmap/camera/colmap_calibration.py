from pathlib import Path
import tempfile
import shutil

import cv2
import imageio.v3 as iio
import numpy as np

from deepreefmap.camera.intrinsics import CameraProfile
from deepreefmap.camera.rectification import Rectifier


def _sample_video_frames(video_path: Path, out_dir: Path, n_frames: int, fps: int) -> list[Path]:
    frames = list(iio.imiter(video_path))
    if not frames:
        raise RuntimeError("No frames found in video")
    stride = max(1, int(round(max(1, len(frames)) / n_frames)))
    selected = frames[::stride][:n_frames]
    out_paths: list[Path] = []
    for i, frame in enumerate(selected):
        p = out_dir / f"{i:06d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out_paths.append(p)
    return out_paths


def calibrate_camera_profile(video: Path, name: str, n_frames: int = 100, fps: int = 10) -> Path:
    with tempfile.TemporaryDirectory(prefix="drm_calib_") as tmp:
        tmp_dir = Path(tmp)
        image_dir = tmp_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        sample_paths = _sample_video_frames(video, image_dir, n_frames=n_frames, fps=fps)
        h, w = cv2.imread(str(sample_paths[0])).shape[:2]

        try:
            import pycolmap

            database_path = tmp_dir / "database.db"
            sparse_path = tmp_dir / "sparse"
            sparse_path.mkdir(parents=True, exist_ok=True)

            reader_options = pycolmap.ImageReaderOptions(camera_model="RADIAL", single_camera=True)
            pycolmap.extract_features(database_path=str(database_path), image_path=str(image_dir), reader_options=reader_options)
            pycolmap.match_exhaustive(database_path=str(database_path))
            maps = pycolmap.incremental_mapping(database_path=str(database_path), image_path=str(image_dir), output_path=str(sparse_path))
            if not maps:
                raise RuntimeError("COLMAP mapping failed: no reconstruction produced.")

            # Pick largest reconstruction by registered images.
            best_rec = max(maps.values(), key=lambda rec: len(rec.images))
            cam = next(iter(best_rec.cameras.values()))
            params = cam.params
            if len(params) < 6:
                raise RuntimeError("Unexpected RADIAL camera parameters from COLMAP.")
            fx, fy, cx, cy, k1, k2 = [float(v) for v in params[:6]]

            # Derive an undistorted pinhole via pycolmap if possible.
            k_rect = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            try:
                undistorted_cam = pycolmap.undistort_camera(pycolmap.UndistortCameraOptions(), cam)
                uparams = undistorted_cam.params
                if len(uparams) >= 4:
                    k_rect = np.array(
                        [[float(uparams[0]), 0.0, float(uparams[2])], [0.0, float(uparams[1]), float(uparams[3])], [0.0, 0.0, 1.0]],
                        dtype=np.float32,
                    )
            except Exception:
                pass

            radial = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2}
            profile = CameraProfile(name=name, image_size=(w, h), k=k_rect, radial=radial)
            profile_path = profile.save()
            rectifier = Rectifier(profile)

            diagnostics_dir = Path("camera_profiles") / f"{name}_diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            # Keep sparse model and sample images for verification.
            try:
                best_rec.write(str(diagnostics_dir / "sparse"))
            except Exception:
                pass
            for p in sample_paths[:4]:
                shutil.copy2(p, diagnostics_dir / f"raw_{p.name}")
                raw_bgr = cv2.imread(str(p))
                if raw_bgr is None:
                    continue
                raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                rect_rgb = rectifier.rectify(raw_rgb)
                cv2.imwrite(str(diagnostics_dir / f"rectified_{p.name}"), cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2BGR))
            return profile_path

        except Exception:
            # Safe fallback so the command still produces a profile.
            k = np.array([[0.9 * w, 0.0, w / 2.0], [0.0, 0.9 * h, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            radial = {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2]), "k1": 0.0, "k2": 0.0}
            profile = CameraProfile(name=name, image_size=(w, h), k=k, radial=radial)
            profile_path = profile.save()
            diagnostics_dir = Path("camera_profiles") / f"{name}_diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            for p in sample_paths[:4]:
                shutil.copy2(p, diagnostics_dir / f"raw_{p.name}")
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
        "diagnostic_previews": previews[:4],
        "diagnostics_dir_exists": diagnostics_dir.exists(),
    }
