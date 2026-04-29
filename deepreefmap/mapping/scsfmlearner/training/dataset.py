from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_image(path: Path) -> np.ndarray:
    image = iio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return image.astype(np.float32)


def _default_intrinsics(height: int, width: int) -> np.ndarray:
    fx = 0.8 * width
    fy = 0.8 * height
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass(frozen=True)
class FramePair:
    sequence_name: str
    image1: Path
    image2: Path
    intrinsics: np.ndarray


class ImageSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: str | Path,
        transform=None,
        skip_frames: int = 1,
        exts: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        include_sequences: set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.skip_frames = max(skip_frames, 1)
        self.exts = {ext.lower() for ext in exts}
        self.include_sequences = include_sequences
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No frame pairs found under {self.root}")

    def _build_samples(self) -> list[FramePair]:
        samples: list[FramePair] = []
        for sequence in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if self.include_sequences is not None and sequence.name not in self.include_sequences:
                continue
            images = [p for p in sorted(sequence.iterdir()) if p.suffix.lower() in self.exts]
            if len(images) < self.skip_frames + 1:
                continue
            intrinsics_path = sequence / "cam.txt"
            intrinsics = None
            if intrinsics_path.exists():
                intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32).reshape(3, 3)
            if intrinsics is None:
                sample_img = _load_image(images[0])
                intrinsics = _default_intrinsics(sample_img.shape[0], sample_img.shape[1])
            for idx in range(0, len(images) - self.skip_frames):
                samples.append(FramePair(sequence.name, images[idx], images[idx + self.skip_frames], intrinsics.copy()))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        img1 = _load_image(sample.image1)
        img2 = _load_image(sample.image2)
        intrinsics = sample.intrinsics.copy()

        if self.transform is not None:
            images, intrinsics = self.transform([img1, img2], intrinsics)
            img1_t, img2_t = images
        else:
            img1_t = torch.from_numpy(np.transpose(img1, (2, 0, 1))).float() / 255.0
            img2_t = torch.from_numpy(np.transpose(img2, (2, 0, 1))).float() / 255.0

        intrinsics_t = torch.from_numpy(intrinsics).float()
        return img1_t, img2_t, intrinsics_t
