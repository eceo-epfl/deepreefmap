from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from PIL import Image


class Compose:
    def __init__(self, transforms: Iterable) -> None:
        self.transforms = list(transforms)

    def __call__(self, images: list[np.ndarray], intrinsics: np.ndarray) -> tuple[list, np.ndarray]:
        for transform in self.transforms:
            images, intrinsics = transform(images, intrinsics)
        return images, intrinsics


class ArrayToTensor:
    def __call__(self, images: list[np.ndarray], intrinsics: np.ndarray) -> tuple[list[torch.Tensor], np.ndarray]:
        tensors: list[torch.Tensor] = []
        for image in images:
            chw = np.transpose(image, (2, 0, 1))
            tensors.append(torch.from_numpy(chw).float() / 255.0)
        return tensors, intrinsics


class Normalize:
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, images: list[torch.Tensor], intrinsics: np.ndarray) -> tuple[list[torch.Tensor], np.ndarray]:
        for image in images:
            for channel, mean, std in zip(image, self.mean, self.std):
                channel.sub_(mean).div_(std)
        return images, intrinsics


class RandomHorizontalFlip:
    def __call__(self, images: list[np.ndarray], intrinsics: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        if random.random() >= 0.5:
            return images, intrinsics
        out_images = [np.copy(np.fliplr(im)) for im in images]
        out_intrinsics = np.copy(intrinsics)
        width = out_images[0].shape[1]
        out_intrinsics[0, 2] = width - out_intrinsics[0, 2]
        return out_images, out_intrinsics


class RandomScaleCrop:
    def __init__(self, max_scale: float = 1.15) -> None:
        self.max_scale = max_scale

    def __call__(self, images: list[np.ndarray], intrinsics: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        h, w, _ = images[0].shape
        x_scale, y_scale = np.random.uniform(1.0, self.max_scale, 2)
        scaled_h, scaled_w = int(h * y_scale), int(w * x_scale)

        out_intrinsics = np.copy(intrinsics)
        out_intrinsics[0] *= x_scale
        out_intrinsics[1] *= y_scale

        scaled_images = [
            np.array(Image.fromarray(image.astype(np.uint8)).resize((scaled_w, scaled_h)), dtype=np.float32) for image in images
        ]
        offset_y = np.random.randint(0, scaled_h - h + 1)
        offset_x = np.random.randint(0, scaled_w - w + 1)
        cropped = [image[offset_y : offset_y + h, offset_x : offset_x + w] for image in scaled_images]

        out_intrinsics[0, 2] -= offset_x
        out_intrinsics[1, 2] -= offset_y
        return cropped, out_intrinsics


class RandomSequencePermutation:
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = min(max(probability, 0.0), 1.0)

    def __call__(self, images: list[np.ndarray], intrinsics: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        if len(images) < 2 or random.random() >= self.probability:
            return images, intrinsics
        permuted = list(images)
        random.shuffle(permuted)
        return permuted, intrinsics
