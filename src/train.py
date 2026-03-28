"""
Training script for ViT-LaneSeg.

Full training loop with:
- AdamW optimizer with cosine annealing + linear warmup
- Mixed-precision (AMP) training
- Gradient clipping
- TensorBoard logging
- Best model checkpointing by validation mIoU
- Early stopping

Usage:
    python -m src.train --config configs/train_config.yaml
    python -m src.train --config configs/train_config.yaml --epochs 50 --batch_size 4
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ViTLaneSeg
from src.data import LaneSegmentationDataset, get_train_transforms, get_val_transforms
from src.losses import FocalDiceLoss
from src.evaluate import SegmentationMetrics, compute_batch_miou


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
    steps_per_epoch: int = 1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine annealing learning rate schedule with linear warmup.

    Warmup: linearly ramp from 0 → base_lr over warmup_epochs.
    Cosine: anneal from base_lr → min_lr over remaining epochs.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return max(current_step / max(warmup_steps, 1), 1e-8)
        else:
            # Cosine annealing
            progress = (current_step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            # Scale between min_lr and 1.0
            return max(min_lr / optimizer.defaults["lr"], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: FocalDiceLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: dict,
    writer: SummaryWriter | None = None,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_focal = 0
    total_dice = 0
    total_miou = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Train]", leave=False)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        global_step = epoch * len(dataloader) + batch_idx

        # Forward pass with mixed precision
        with autocast("cuda", enabled=config["training"].get("amp", True)):
            logits = model(images)
            loss, loss_dict = criterion(logits, masks)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        grad_clip = config["training"].get("grad_clip", 0)
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics
        with torch.no_grad():
            batch_miou = compute_batch_miou(logits, masks, config["model"]["num_classes"])

        total_loss += loss_dict["total_loss"]
        total_focal += loss_dict["focal_loss"]
        total_dice += loss_dict["dice_loss"]
        total_miou += batch_miou
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss']:.4f}",
            "mIoU": f"{batch_miou:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        # TensorBoard logging
        log_every = config["logging"].get("log_every", 10)
        if writer and batch_idx % log_every == 0:
            writer.add_scalar("train/total_loss", loss_dict["total_loss"], global_step)
            writer.add_scalar("train/focal_loss", loss_dict["focal_loss"], global_step)
            writer.add_scalar("train/dice_loss", loss_dict["dice_loss"], global_step)
            writer.add_scalar("train/batch_miou", batch_miou, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    avg_metrics = {
        "loss": total_loss / num_batches,
        "focal_loss": total_focal / num_batches,
        "dice_loss": total_dice / num_batches,
        "miou": total_miou / num_batches,
    }

    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: FocalDiceLoss,
    device: torch.device,
    epoch: int,
    config: dict,
) -> dict:
    """Run validation."""
    model.eval()

    num_classes = config["model"]["num_classes"]
    class_names = ["background", "dashed", "solid"]
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)

    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        with autocast("cuda", enabled=config["training"].get("amp", True)):
            logits = model(images)
            loss, _ = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

        total_loss += loss.item()
        num_batches += 1

    results = metrics.compute()
    results["loss"] = total_loss / num_batches

    # Print report
    metrics.print_report(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ViT-LaneSeg")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                       help="Path to config YAML file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # =============================
    # Create model
    # =============================
    model = ViTLaneSeg.from_config(config)
    model = model.to(device)

    param_info = model.get_param_count()
    print(f"[Train] Model parameters: {param_info['total_M']}")
    print(f"  Encoder: {param_info['encoder']:,}")
    print(f"  Decoder: {param_info['decoder']:,}")

    # =============================
    # Create datasets
    # =============================
    img_size = tuple(config["model"]["img_size"])
    train_transforms = get_train_transforms(img_size)
    val_transforms = get_val_transforms(img_size)

    train_dataset = LaneSegmentationDataset(
        image_dir=config["data"]["train_image_dir"],
        mask_dir=config["data"]["train_mask_dir"],
        transform=train_transforms,
        img_size=img_size,
        num_classes=config["model"]["num_classes"],
    )

    val_dataset = LaneSegmentationDataset(
        image_dir=config["data"]["val_image_dir"],
        mask_dir=config["data"]["val_mask_dir"],
        transform=val_transforms,
        img_size=img_size,
        num_classes=config["model"]["num_classes"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )

    # =============================
    # Loss function
    # =============================
    class_weights = None
    if config["loss"].get("use_class_weights", True):
        class_weights = train_dataset.compute_class_weights()

    criterion = FocalDiceLoss(
        focal_weight=config["loss"].get("focal_weight", 1.0),
        dice_weight=config["loss"].get("dice_weight", 1.0),
        alpha=class_weights,
        gamma=config["loss"].get("focal_gamma", 2.0),
    )

    # =============================
    # Optimizer & scheduler
    # =============================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.05),
        betas=(0.9, 0.999),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=config["training"].get("warmup_epochs", 5),
        total_epochs=config["training"]["epochs"],
        min_lr=config["training"].get("min_lr", 1e-6),
        steps_per_epoch=len(train_loader),
    )

    scaler = GradScaler(enabled=config["training"].get("amp", True))

    # =============================
    # Resume from checkpoint
    # =============================
    start_epoch = 0
    best_miou = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)
        print(f"[Train] Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")

    # =============================
    # TensorBoard
    # =============================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config["logging"].get("log_dir", "runs")) / f"vit_lane_{timestamp}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[Train] TensorBoard logs: {log_dir}")

    # Checkpoint directory
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # =============================
    # Training loop
    # =============================
    patience = config["training"].get("early_stopping_patience", 15)
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f"Starting training for {config['training']['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, config, writer,
        )

        # Validate
        val_every = config["logging"].get("val_every", 1)
        if (epoch + 1) % val_every == 0:
            val_results = validate(
                model, val_loader, criterion, device, epoch, config,
            )
            val_miou = val_results["miou"]
            val_loss = val_results["loss"]

            # TensorBoard
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/miou", val_miou, epoch)
            for cls_name, iou in val_results["per_class_iou"].items():
                writer.add_scalar(f"val/iou_{cls_name}", iou, epoch)

            # Check for improvement
            if val_miou > best_miou:
                best_miou = val_miou
                epochs_without_improvement = 0

                # Save best model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_miou": best_miou,
                    "config": config,
                }, ckpt_dir / "best_model.pth")

                print(f"  ★ New best mIoU: {best_miou:.4f} — saved checkpoint")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n[Train] Early stopping: no improvement for {patience} epochs")
                break

        # Periodic checkpoint
        save_every = config["training"].get("save_every", 5)
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_miou": best_miou,
                "config": config,
            }, ckpt_dir / f"epoch_{epoch:03d}.pth")

        # Epoch summary
        elapsed = time.time() - epoch_start
        print(
            f"  Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train mIoU: {train_metrics['miou']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Log epoch-level train metrics
        writer.add_scalar("train_epoch/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train_epoch/miou", train_metrics["miou"], epoch)

    # =============================
    # Save final model
    # =============================
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_miou": best_miou,
    }, ckpt_dir / "final_model.pth")

    writer.close()

    print(f"\n{'='*60}")
    print(f"Training complete! Best mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {ckpt_dir / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
