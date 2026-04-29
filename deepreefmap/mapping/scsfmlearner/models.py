from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


def _euler_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    zeros = torch.zeros_like(cx)
    ones = torch.ones_like(cx)

    rot_x = torch.stack(
        [ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
        dim=1,
    ).reshape(-1, 3, 3)
    rot_y = torch.stack(
        [cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
        dim=1,
    ).reshape(-1, 3, 3)
    rot_z = torch.stack(
        [cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
        dim=1,
    ).reshape(-1, 3, 3)
    return rot_x @ rot_y @ rot_z


def pose_vec_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """Convert 6-DoF vectors [tx,ty,tz,rx,ry,rz] to 4x4 transforms."""
    t = pose[:, :3]
    r = pose[:, 3:]
    rot = _euler_to_matrix(r)
    transform = torch.eye(4, dtype=pose.dtype, device=pose.device).unsqueeze(0).repeat(pose.shape[0], 1, 1)
    transform[:, :3, :3] = rot
    transform[:, :3, 3] = t
    return transform


class ResnetEncoder(nn.Module):
    def __init__(self, num_layers: int = 18, pretrained: bool = True, num_input_images: int = 1) -> None:
        super().__init__()
        if num_layers == 18:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            encoder = models.resnet18(weights=weights)
            self.num_ch_enc = [64, 64, 128, 256, 512]
        elif num_layers == 50:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            encoder = models.resnet50(weights=weights)
            self.num_ch_enc = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError("num_layers must be one of {18, 50}")

        if num_input_images > 1:
            old = encoder.conv1
            encoder.conv1 = nn.Conv2d(
                in_channels=3 * num_input_images,
                out_channels=old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained:
                encoder.conv1.weight.data = old.weight.data.repeat(1, num_input_images, 1, 1) / num_input_images

        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        features.append(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        features.append(x)
        x = self.encoder.layer2(x)
        features.append(x)
        x = self.encoder.layer3(x)
        features.append(x)
        x = self.encoder.layer4(x)
        features.append(x)
        return features


class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(Conv3x3(in_channels, out_channels), nn.ELU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc: Sequence[int],
        scales: Sequence[int] = (0, 1, 2, 3),
        alpha: float = 10.0,
        beta: float = 0.01,
    ) -> None:
        super().__init__()
        self.scales = set(scales)
        self.alpha = alpha
        self.beta = beta
        self.num_ch_dec = [16, 32, 64, 128, 256]

        self.upconv0 = nn.ModuleList()
        self.upconv1 = nn.ModuleList()
        self.dispconv = nn.ModuleDict()
        for i in range(4, -1, -1):
            in_ch = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            self.upconv0.append(ConvBlock(in_ch, self.num_ch_dec[i]))
            in_ch_1 = self.num_ch_dec[i] + (num_ch_enc[i - 1] if i > 0 else 0)
            self.upconv1.append(ConvBlock(in_ch_1, self.num_ch_dec[i]))
            if i in self.scales:
                self.dispconv[str(i)] = Conv3x3(self.num_ch_dec[i], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = input_features[-1]
        for idx, i in enumerate(range(4, -1, -1)):
            x = self.upconv0[idx](x)
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            if i > 0:
                x = torch.cat([x, input_features[i - 1]], dim=1)
            x = self.upconv1[idx](x)
            if i in self.scales:
                disp = self.alpha * self.sigmoid(self.dispconv[str(i)](x)) + self.beta
                outputs.append(disp)
        outputs.reverse()
        return outputs


class DispResNet(nn.Module):
    def __init__(self, num_layers: int = 18, pretrained: bool = True, scales: Sequence[int] = (0, 1, 2, 3)) -> None:
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=1)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=scales)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        disps = self.decoder(self.encoder(image))
        if self.training:
            return disps
        return disps[0]


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc: Sequence[int]) -> None:
        super().__init__()
        self.squeeze = nn.Conv2d(num_ch_enc[-1], 256, kernel_size=1)
        self.pose0 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pose1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pose2 = nn.Conv2d(256, 6, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        x = self.relu(self.squeeze(features[-1]))
        x = self.relu(self.pose0(x))
        x = self.relu(self.pose1(x))
        x = self.pose2(x).mean(dim=[2, 3])
        return 0.01 * x


class PoseResNet(nn.Module):
    def __init__(self, num_layers: int = 18, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=2)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(torch.cat([img1, img2], dim=1)))
