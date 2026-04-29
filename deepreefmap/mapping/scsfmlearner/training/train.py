from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from deepreefmap.mapping.scsfmlearner.models import DispResNet, PoseResNet
from deepreefmap.mapping.scsfmlearner.training.dataset import ImageSequenceDataset
from deepreefmap.mapping.scsfmlearner.training.losses import compute_training_loss
from deepreefmap.mapping.scsfmlearner.training.transforms import ArrayToTensor, Compose, Normalize, RandomHorizontalFlip, RandomScaleCrop


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    run_name: str = "scsfmlearner"
    epochs: int = 20
    batch_size: int = 4
    workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    train_split: float = 0.9
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
    device: str | None = None


def _build_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_transform = Compose(
        [
            RandomHorizontalFlip(),
            RandomScaleCrop(),
            ArrayToTensor(),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ]
    )
    val_transform = Compose([ArrayToTensor(), Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))])
    base_dataset = ImageSequenceDataset(config.data_root, transform=None)
    train_len = max(1, int(len(base_dataset) * config.train_split))
    train_len = min(train_len, len(base_dataset) - 1) if len(base_dataset) > 1 else 1
    indices = torch.randperm(len(base_dataset), generator=torch.Generator().manual_seed(0)).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:] or indices[:1]

    train_dataset = ImageSequenceDataset(config.data_root, transform=train_transform)
    val_dataset = ImageSequenceDataset(config.data_root, transform=val_transform)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _epoch_loss(
    loader: DataLoader,
    disp_net: DispResNet,
    pose_net: PoseResNet,
    optimizer: torch.optim.Optimizer | None,
    config: TrainConfig,
    device: torch.device,
) -> float:
    running = 0.0
    steps = 0
    train_mode = optimizer is not None
    disp_net.train(mode=train_mode)
    pose_net.train(mode=train_mode)

    for img1, img2, intrinsics in loader:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        intrinsics = intrinsics.to(device, non_blocking=True)

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
        running += float(loss.detach().cpu())
        steps += 1
    return running / max(steps, 1)


def train(config: TrainConfig) -> Path:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _build_loaders(config)
    disp_net = DispResNet(num_layers=config.resnet_layers, pretrained=config.pretrained).to(device)
    pose_net = PoseResNet(num_layers=18, pretrained=config.pretrained).to(device)

    optimizer = torch.optim.Adam(
        list(disp_net.parameters()) + list(pose_net.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch_loss(train_loader, disp_net, pose_net, optimizer, config, device)
        with torch.no_grad():
            val_loss = _epoch_loss(val_loader, disp_net, pose_net, None, config, device)
        print(f"[{epoch}/{config.epochs}] train={train_loss:.4f} val={val_loss:.4f}")

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
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train compact SC-SfMLearner on simple image sequences.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing sequence subfolders")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--run-name", default="scsfmlearner")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.lr,
    )
    train(cfg)
