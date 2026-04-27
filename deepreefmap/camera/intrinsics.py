from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


CAMERA_PROFILE_DIR = Path("camera_profiles")


@dataclass
class CameraProfile:
    name: str
    image_size: tuple[int, int]
    k: np.ndarray
    radial: dict[str, float]

    @classmethod
    def load(cls, name: str) -> "CameraProfile":
        path = CAMERA_PROFILE_DIR / f"{name}.json"
        data = json.loads(path.read_text())
        k = np.array(data["rectified_pinhole"]["K"], dtype=np.float32)
        size = tuple(data["rectified_pinhole"]["image_size"])
        return cls(
            name=data["name"],
            image_size=(int(size[0]), int(size[1])),
            k=k,
            radial=data["distorted"]["params"],
        )

    def save(self) -> Path:
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
        path.write_text(json.dumps(payload, indent=2))
        return path
