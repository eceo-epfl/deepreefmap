from __future__ import annotations

import torch
from torch.nn import functional as F


def _euler_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    x, y, z = angles[:, 0], angles[:, 1], angles[:, 2]
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    zeros = torch.zeros_like(cx)
    ones = torch.ones_like(cx)

    rot_x = torch.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=1).reshape(-1, 3, 3)
    rot_y = torch.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=1).reshape(-1, 3, 3)
    rot_z = torch.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=1).reshape(-1, 3, 3)
    return rot_x @ rot_y @ rot_z


def pose_vec2mat(pose: torch.Tensor) -> torch.Tensor:
    translation = pose[:, :3].unsqueeze(-1)
    rotation = _euler_to_matrix(pose[:, 3:])
    return torch.cat([rotation, translation], dim=2)


def _pixel_grid(batch_size: int, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0).reshape(1, 3, -1)
    return grid.repeat(batch_size, 1, 1)


def pixel2cam(depth: torch.Tensor, intrinsics_inv: torch.Tensor) -> torch.Tensor:
    b, _, h, w = depth.shape
    pixels = _pixel_grid(b, h, w, depth.dtype, depth.device)
    cam = (intrinsics_inv @ pixels).reshape(b, 3, h, w)
    return cam * depth


def cam2pixel(cam_coords: torch.Tensor, proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = cam_coords.shape
    flat = cam_coords.reshape(b, 3, -1)
    pcoords = proj[:, :, :3] @ flat + proj[:, :, 3:].contiguous()
    x = pcoords[:, 0]
    y = pcoords[:, 1]
    z = pcoords[:, 2].clamp(min=1e-3)

    x_norm = 2.0 * (x / z) / max(w - 1, 1) - 1.0
    y_norm = 2.0 * (y / z) / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=2).reshape(b, h, w, 2)
    return grid, z.reshape(b, 1, h, w)


def inverse_warp(
    src_img: torch.Tensor,
    tgt_depth: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor]:
    cam_coords = pixel2cam(tgt_depth, intrinsics.inverse())
    proj = intrinsics @ pose_vec2mat(pose)
    grid, _ = cam2pixel(cam_coords, proj)
    warped = F.grid_sample(src_img, grid, padding_mode=padding_mode, align_corners=False)
    valid_mask = (grid.abs().amax(dim=-1, keepdim=True) <= 1.0).permute(0, 3, 1, 2).float()
    return warped, valid_mask


def inverse_warp_depth(
    src_img: torch.Tensor,
    tgt_depth: torch.Tensor,
    src_depth: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    padding_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cam_coords = pixel2cam(tgt_depth, intrinsics.inverse())
    proj = intrinsics @ pose_vec2mat(pose)
    grid, computed_depth = cam2pixel(cam_coords, proj)
    warped = F.grid_sample(src_img, grid, padding_mode=padding_mode, align_corners=False)
    projected_depth = F.grid_sample(src_depth, grid, padding_mode=padding_mode, align_corners=False)
    valid_mask = (grid.abs().amax(dim=-1, keepdim=True) <= 1.0).permute(0, 3, 1, 2).float()
    return warped, valid_mask, projected_depth, computed_depth
