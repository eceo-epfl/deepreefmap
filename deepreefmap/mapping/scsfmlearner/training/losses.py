from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from deepreefmap.mapping.scsfmlearner.training.warping import inverse_warp_depth


class SSIM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.c1 = 0.01**2
        self.c2 = 0.03**2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        n = (2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)
        d = (mu_x**2 + mu_y**2 + self.c1) * (sigma_x + sigma_y + self.c2)
        return torch.clamp((1 - n / d) * 0.5, 0, 1)


def _masked_mean(diff: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(diff)
    denom = expanded.sum().clamp(min=1.0)
    return (diff * expanded).sum() / denom


def photometric_loss(
    target_img: torch.Tensor,
    source_img: torch.Tensor,
    warped_img: torch.Tensor,
    valid_mask: torch.Tensor,
    ssim_module: SSIM,
    with_ssim: bool = True,
    ssim_weight: float = 0.85,
    auto_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    diff = (target_img - warped_img).abs()
    if with_ssim:
        diff = (1.0 - ssim_weight) * diff + ssim_weight * ssim_module(target_img, warped_img)
    if auto_mask:
        identity = (target_img - source_img).abs().mean(dim=1, keepdim=True)
        reproj = diff.mean(dim=1, keepdim=True)
        valid_mask = valid_mask * (reproj < identity).float()
    return _masked_mean(diff, valid_mask), valid_mask


def geometry_consistency_loss(
    projected_depth: torch.Tensor,
    computed_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    diff = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth + 1e-7)).clamp(0, 1)
    return _masked_mean(diff, valid_mask)


def smooth_loss(disp: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    norm_disp = disp / (disp.mean(dim=[2, 3], keepdim=True) + 1e-7)
    grad_disp_x = (norm_disp[:, :, :, :-1] - norm_disp[:, :, :, 1:]).abs()
    grad_disp_y = (norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :]).abs()
    grad_img_x = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean(dim=1, keepdim=True)
    grad_img_y = (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean(dim=1, keepdim=True)
    return (grad_disp_x * torch.exp(-grad_img_x)).mean() + (grad_disp_y * torch.exp(-grad_img_y)).mean()


def _as_list(tensor_or_list: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(tensor_or_list, torch.Tensor):
        return [tensor_or_list]
    return list(tensor_or_list)


def compute_training_loss(
    img1: torch.Tensor,
    img2: torch.Tensor,
    intrinsics: torch.Tensor,
    disp_net: nn.Module,
    pose_net: nn.Module,
    num_scales: int = 1,
    photo_weight: float = 1.0,
    smooth_weight: float = 0.1,
    geometry_weight: float = 0.5,
    with_ssim: bool = True,
    auto_mask: bool = False,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ssim = SSIM().to(img1.device)
    disp1_scales = _as_list(disp_net(img1))
    disp2_scales = _as_list(disp_net(img2))
    pose_12 = pose_net(img1, img2)
    pose_21 = pose_net(img2, img1)

    scales = min(num_scales, len(disp1_scales), len(disp2_scales))
    photo = torch.tensor(0.0, device=img1.device)
    smooth = torch.tensor(0.0, device=img1.device)
    geometry = torch.tensor(0.0, device=img1.device)

    h, w = img1.shape[-2:]
    for s in range(scales):
        disp1 = disp1_scales[s]
        disp2 = disp2_scales[s]
        if s > 0:
            disp1 = F.interpolate(disp1, size=(h, w), mode="nearest")
            disp2 = F.interpolate(disp2, size=(h, w), mode="nearest")
        depth1 = 1.0 / disp1.clamp(min=1e-6)
        depth2 = 1.0 / disp2.clamp(min=1e-6)

        warped2, mask12, proj2, comp2 = inverse_warp_depth(img2, depth1, depth2, pose_12, intrinsics, padding_mode)
        warped1, mask21, proj1, comp1 = inverse_warp_depth(img1, depth2, depth1, pose_21, intrinsics, padding_mode)

        photo12, mask12 = photometric_loss(img1, img2, warped2, mask12, ssim, with_ssim, auto_mask=auto_mask)
        photo21, mask21 = photometric_loss(img2, img1, warped1, mask21, ssim, with_ssim, auto_mask=auto_mask)
        photo = photo + photo12 + photo21
        geometry = geometry + geometry_consistency_loss(proj2, comp2, mask12) + geometry_consistency_loss(proj1, comp1, mask21)
        smooth = smooth + smooth_loss(disp1, img1) + smooth_loss(disp2, img2)

    total = photo_weight * photo + smooth_weight * smooth + geometry_weight * geometry
    return total, {"photo": photo.detach(), "smooth": smooth.detach(), "geometry": geometry.detach()}
