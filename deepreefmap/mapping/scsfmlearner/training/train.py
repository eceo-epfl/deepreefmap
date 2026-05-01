from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

from deepreefmap.mapping.scsfmlearner.models import DispResNet, PoseResNet
from deepreefmap.mapping.scsfmlearner.training.dataset import ImageSequenceDataset
from deepreefmap.mapping.scsfmlearner.training.losses import compute_training_loss
from deepreefmap.mapping.scsfmlearner.training.transforms import (
    ArrayToTensor,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomScaleCrop,
    RandomSequencePermutation,
)


@dataclass
class TrainConfig:
    train_data_root: str
    eval_data_root: str
    output_dir: str
    run_name: str = "scsfmlearner"
    epochs: int = 20
    batch_size: int = 32
    workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    skip_frames: int = 3
    train_steps_per_epoch: int = 5000
    eval_steps_per_epoch: int = 1000
    sequence_permutation_prob: float = 0.5
    pretrained: bool = True
    resnet_layers: int = 18
    num_scales: int = 1
    photo_weight: float = 1.0
    smooth_weight: float = 0.1
    geometry_weight: float = 0.5
    with_ssim: bool = True
    auto_mask: bool = False
    padding_mode: str = "zeros"
    save_every: int = 1
    log_every: int = 10
    dataset_refresh_every: int = 5
    device: str | None = None


def _build_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_transform = Compose(
        [
            RandomHorizontalFlip(),
            RandomScaleCrop(),
            RandomSequencePermutation(probability=config.sequence_permutation_prob),
            ArrayToTensor(),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ]
    )
    val_transform = Compose([ArrayToTensor(), Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])
    train_dataset = ImageSequenceDataset(
        config.train_data_root,
        transform=train_transform,
        skip_frames=config.skip_frames,
    )
    val_dataset = ImageSequenceDataset(
        config.eval_data_root,
        transform=val_transform,
        skip_frames=config.skip_frames,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _step_loss(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    disp_net: DispResNet,
    pose_net: PoseResNet,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
) -> tuple[float, torch.Tensor]:
    train_mode = optimizer is not None
    img1, img2, intrinsics = batch
    loss, _ = compute_training_loss(
        img1,
        img2,
        intrinsics,
        disp_net=disp_net,
        pose_net=pose_net,
        num_scales=config.num_scales,
        photo_weight=config.photo_weight,
        smooth_weight=config.smooth_weight,
        geometry_weight=config.geometry_weight,
        with_ssim=config.with_ssim,
        auto_mask=config.auto_mask,
        padding_mode=config.padding_mode,
    )
    if train_mode:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu()), img1


def _next_batch(loader_iter: iter, loader: DataLoader, device: torch.device) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], iter]:
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    img1, img2, intrinsics = batch
    return (
        img1.to(device, non_blocking=True),
        img2.to(device, non_blocking=True),
        intrinsics.to(device, non_blocking=True),
    ), loader_iter


def _make_preview_tensors(image: torch.Tensor, disp_net: DispResNet) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor((0.45, 0.45, 0.45), device=image.device).view(1, 3, 1, 1)
    std = torch.tensor((0.225, 0.225, 0.225), device=image.device).view(1, 3, 1, 1)
    rgb = (image * std + mean).clamp(0.0, 1.0)
    with torch.no_grad():
        disp = disp_net(image)
        if isinstance(disp, list):
            disp = disp[0]
        depth = (1.0 / disp.clamp(min=1e-6)).detach()
        depth = depth - depth.amin(dim=(2, 3), keepdim=True)
        depth = depth / depth.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        depth_rgb = depth.repeat(1, 3, 1, 1)
    return rgb, depth_rgb


def _preview_for_wandb(image: torch.Tensor, disp_net: DispResNet, split: str) -> tuple[list, list]:
    rgb, depth_rgb = _make_preview_tensors(image, disp_net)
    rgb_preview = rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
    depth_preview = depth_rgb.detach().cpu().permute(0, 2, 3, 1).numpy()
    rgb_images = [wandb.Image(sample, caption=f"{split}_rgb_{idx}") for idx, sample in enumerate(rgb_preview)]
    depth_images = [wandb.Image(sample, caption=f"{split}_depth_{idx}") for idx, sample in enumerate(depth_preview)]
    return rgb_images, depth_images


def train(config: TrainConfig) -> Path:
    if wandb is None:
        raise RuntimeError("wandb is required for training logging but is not installed.")
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = None
    val_loader = None
    disp_net = DispResNet(num_layers=config.resnet_layers, pretrained=config.pretrained).to(device)
    pose_net = PoseResNet(num_layers=18, pretrained=config.pretrained).to(device)

    optimizer = torch.optim.Adam(
        list(disp_net.parameters()) + list(pose_net.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    train_iter = None
    wandb.init(project="deepreefmap-scsfmlearner", name=config.run_name, config=config.__dict__, dir=str(output_dir))
    try:
        for epoch in range(1, config.epochs + 1):
            refresh_every = max(config.dataset_refresh_every, 1)
            if train_loader is None or val_loader is None or (epoch - 1) % refresh_every == 0:
                train_loader, val_loader = _build_loaders(config)
                train_iter = iter(train_loader)
                print(
                    f"Refreshed datasets at epoch {epoch}: "
                    f"train_pairs={len(train_loader.dataset)} eval_pairs={len(val_loader.dataset)}"
                )

            disp_net.train()
            pose_net.train()
            train_running = 0.0
            last_train_img = None
            for _ in range(config.train_steps_per_epoch):
                assert train_loader is not None and train_iter is not None
                batch, train_iter = _next_batch(train_iter, train_loader, device)
                loss_value, last_train_img = _step_loss(batch, disp_net, pose_net, optimizer, config)
                train_running += loss_value
            train_loss = train_running / max(config.train_steps_per_epoch, 1)

            disp_net.eval()
            pose_net.eval()
            val_running = 0.0
            val_steps = 0
            last_eval_img = None
            assert val_loader is not None
            val_iter = iter(val_loader)
            with torch.no_grad():
                for _ in range(config.eval_steps_per_epoch):
                    try:
                        raw_batch = next(val_iter)
                    except StopIteration:
                        break
                    img1, img2, intrinsics = raw_batch
                    batch = (
                        img1.to(device, non_blocking=True),
                        img2.to(device, non_blocking=True),
                        intrinsics.to(device, non_blocking=True),
                    )
                    loss_value, last_eval_img = _step_loss(batch, disp_net, pose_net, None, config)
                    val_running += loss_value
                    val_steps += 1
            if val_steps == 0:
                raise RuntimeError("Evaluation dataset yielded no batches. Check eval_data_root and batch_size.")
            val_loss = val_running / val_steps
            print(f"[{epoch}/{config.epochs}] train={train_loss:.4f} val={val_loss:.4f}")
            metrics = {"loss/train": train_loss, "loss/val": val_loss, "epoch": epoch}
            if epoch % config.log_every == 0 or epoch == 1:
                preview_count = min(4, config.batch_size)
                if last_train_img is not None:
                    train_rgb, train_depth = _preview_for_wandb(
                        last_train_img[: min(preview_count, last_train_img.shape[0])], disp_net, split="train"
                    )
                    metrics["preview/train_rgb"] = train_rgb
                    metrics["preview/train_depth"] = train_depth
                if last_eval_img is not None:
                    eval_rgb, eval_depth = _preview_for_wandb(
                        last_eval_img[: min(preview_count, last_eval_img.shape[0])], disp_net, split="eval"
                    )
                    metrics["preview/eval_rgb"] = eval_rgb
                    metrics["preview/eval_depth"] = eval_depth
            wandb.log(metrics, step=epoch)

            checkpoint = {
                "epoch": epoch,
                "config": config.__dict__,
                "disp_state_dict": disp_net.state_dict(),
                "pose_state_dict": pose_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            if epoch % config.save_every == 0:
                torch.save(checkpoint, output_dir / f"checkpoint_{epoch:04d}.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint, output_dir / "best.pt")
    finally:
        wandb.finish()
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train compact SC-SfMLearner on simple image sequences.")
    parser.add_argument("--train-data-root", required=True, help="Training dataset root containing sequence subfolders")
    parser.add_argument("--eval-data-root", required=True, help="Evaluation dataset root containing sequence subfolders")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--run-name", default="scsfmlearner")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip-frames", type=int, default=3)
    parser.add_argument("--train-steps-per-epoch", type=int, default=5000)
    parser.add_argument("--eval-steps-per-epoch", type=int, default=1000)
    parser.add_argument("--sequence-permutation-prob", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dataset-refresh-every", type=int, default=5)
    args = parser.parse_args()

    cfg = TrainConfig(
        train_data_root=args.train_data_root,
        eval_data_root=args.eval_data_root,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.lr,
        skip_frames=args.skip_frames,
        train_steps_per_epoch=args.train_steps_per_epoch,
        eval_steps_per_epoch=args.eval_steps_per_epoch,
        sequence_permutation_prob=args.sequence_permutation_prob,
        log_every=args.log_every,
        dataset_refresh_every=args.dataset_refresh_every,
    )
    train(cfg)
