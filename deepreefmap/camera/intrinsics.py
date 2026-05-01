from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import json
import re

import numpy as np


CAMERA_PROFILE_DIR = Path("camera_profiles")
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class CameraProfile:
    name: str
    image_size: tuple[int, int]
    k: np.ndarray
    radial: dict[str, float]
    diagnostics: dict[str, object] | None = None

    @classmethod
    def load(cls, name: str) -> "CameraProfile":
        validate_profile_name(name)
        path = CAMERA_PROFILE_DIR / f"{name}.json"
        data = json.loads(_read_profile_text(name, path))
        k = np.array(data["rectified_pinhole"]["K"], dtype=np.float32)
        size = tuple(data["rectified_pinhole"]["image_size"])
        return cls(
            name=data["name"],
            image_size=(int(size[0]), int(size[1])),
            k=k,
            radial=data["distorted"]["params"],
            diagnostics=data.get("diagnostics"),
        )

    def save(self) -> Path:
        validate_profile_name(self.name)
        CAMERA_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = CAMERA_PROFILE_DIR / f"{self.name}.json"
        payload = {
            "name": self.name,
            "source": "colmap_radial_v1",
            "distorted": {"model": "RADIAL", "params": self.radial},
            "rectified_pinhole": {
                "image_size": [int(self.image_size[0]), int(self.image_size[1])],
                "K": self.k.tolist(),
            },
        }
        if self.diagnostics is not None:
            payload["diagnostics"] = self.diagnostics
        path.write_text(json.dumps(payload, indent=2))
        return path


def validate_profile_name(name: str) -> None:
    if not _PROFILE_NAME_RE.fullmatch(name):
        raise ValueError("Camera profile names may only contain letters, numbers, underscores, and hyphens.")


def available_profile_names() -> list[str]:
    names = set[str]()
    if CAMERA_PROFILE_DIR.exists():
        names.update(p.stem for p in CAMERA_PROFILE_DIR.glob("*.json"))
    resource_dir = resources.files("deepreefmap.resources").joinpath("camera_profiles")
    if resource_dir.is_dir():
        names.update(p.name.removesuffix(".json") for p in resource_dir.iterdir() if p.name.endswith(".json"))
    return sorted(names)


def scale_intrinsics(
    k: np.ndarray,
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Rescale a 3x3 pinhole intrinsics matrix from ``original_size`` to ``target_size``.

    Both sizes are ``(width, height)``. The bottom row is reset to ``[0, 0, 1]`` to
    suppress accumulated float drift across repeated rescalings.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    scaled = k.astype(np.float32).copy()
    scaled[0, :] *= float(target_w) / max(float(orig_w), 1.0)
    scaled[1, :] *= float(target_h) / max(float(orig_h), 1.0)
    scaled[2] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return scaled


def _read_profile_text(name: str, path: Path) -> str:
    if path.exists():
        return path.read_text()
    resource = resources.files("deepreefmap.resources").joinpath("camera_profiles", f"{name}.json")
    if resource.is_file():
        return resource.read_text()
    raise FileNotFoundError(f"Camera profile not found: {path}")
