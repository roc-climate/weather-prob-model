"""
Training script for the weather probability model.

Phase 1: Monthly data, single lead time (1 month ~ 4 weeks ahead).
Targets: t2m (Gaussian head + CRPS loss), tp (Quantile head + pinball loss).

Usage:
  python -m training.train --config ../configs/config.yaml
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import WeatherDataset
from data.normalization import compute_statistics, load_statistics, save_statistics
from models.weather_model import WeatherProbModel
from losses.crps_loss import GaussianCRPSLoss
from losses.quantile_loss import QuantilePinballLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train weather probability model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict, device: torch.device) -> WeatherProbModel:
    """Build model from config."""
    model_cfg = config["model"]
    model = WeatherProbModel(
        n_atmos_vars=6,   # msl, u10, v10, tisr, ssr, str
        n_slow_vars=4,    # sst, swvl1, sd, siconc
        n_indices=6,      # RMM1, RMM2, Nino3.4, NAO, AO, doy
        atmos_encoder_kwargs=model_cfg.get("atmos_encoder", {}),
        slow_encoder_kwargs=model_cfg.get("slow_encoder", {}),
        cross_attn_kwargs=model_cfg.get("cross_attention", {}),
        index_embed_kwargs=model_cfg.get("index_embedding", {}),
        gaussian_head_kwargs=model_cfg.get("prob_heads", {}).get("t2m", {}),
        quantile_head_kwargs=model_cfg.get("prob_heads", {}).get("tp", {}),
    )
    model = model.to(device)
    return model


def setup_losses(config: dict) -> Dict[str, nn.Module]:
    """Setup loss functions."""
    loss_cfg = config["training"].get("loss_weights", {})
    use_lat_weight = config["training"].get("use_latitude_weight", True)

    return {
        "t2m": GaussianCRPSLoss(use_latitude_weight=use_lat_weight, n_lat=121),
        "tp": QuantilePinballLoss(use_latitude_weight=use_lat_weight, n_lat=121),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    losses: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    loss_weights: dict,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {"total": 0.0, "t2m": 0.0, "tp": 0.0}
    n_batches = 0

    for batch in dataloader:
        x_atmos = batch["x_atmos"].to(device)
        x_slow = batch["x_slow"].to(device)
        y_t2m = batch["y_t2m"].to(device)
        y_tp = batch["y_tp"].to(device)

        # Phase 1: placeholder indices (will be real in Phase 2)
        B = x_atmos.shape[0]
        x_index = torch.zeros(B, 6, device=device)

        # Remove target variables from input
        x_atmos_input = x_atmos  # Already excludes t2m, tp via dataset._select_vars

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(x_atmos_input, x_slow, x_index)
                loss_t2m = losses["t2m"](pred["t2m"], y_t2m)
                loss_tp = losses["tp"](pred["tp"], y_tp)
                w_t2m = loss_weights.get("t2m", 1.0)
                w_tp = loss_weights.get("tp", 0.5)
                total_loss = w_t2m * loss_t2m + w_tp * loss_tp
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x_atmos_input, x_slow, x_index)
            loss_t2m = losses["t2m"](pred["t2m"], y_t2m)
            loss_tp = losses["tp"](pred["tp"], y_tp)
            w_t2m = loss_weights.get("t2m", 1.0)
            w_tp = loss_weights.get("tp", 0.5)
            total_loss = w_t2m * loss_t2m + w_tp * loss_tp
            total_loss.backward()
            optimizer.step()

        epoch_losses["total"] += total_loss.item()
        epoch_losses["t2m"] += loss_t2m.item()
        epoch_losses["tp"] += loss_tp.item()
        n_batches += 1

    for k in epoch_losses:
        epoch_losses[k] /= max(n_batches, 1)

    return epoch_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    losses: Dict[str, nn.Module],
    loss_weights: dict,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    val_losses = {"total": 0.0, "t2m": 0.0, "tp": 0.0}
    n_batches = 0

    for batch in dataloader:
        x_atmos = batch["x_atmos"].to(device)
        x_slow = batch["x_slow"].to(device)
        y_t2m = batch["y_t2m"].to(device)
        y_tp = batch["y_tp"].to(device)

        B = x_atmos.shape[0]
        x_index = torch.zeros(B, 6, device=device)

        x_atmos_input = x_atmos

        pred = model(x_atmos_input, x_slow, x_index)
        loss_t2m = losses["t2m"](pred["t2m"], y_t2m)
        loss_tp = losses["tp"](pred["tp"], y_tp)
        w_t2m = loss_weights.get("t2m", 1.0)
        w_tp = loss_weights.get("tp", 0.5)
        total_loss = w_t2m * loss_t2m + w_tp * loss_tp

        val_losses["total"] += total_loss.item()
        val_losses["t2m"] += loss_t2m.item()
        val_losses["tp"] += loss_tp.item()
        n_batches += 1

    for k in val_losses:
        val_losses[k] /= max(n_batches, 1)

    return val_losses


def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup
    device = torch.device(args.device or config["hardware"]["device"])
    seed = config["hardware"]["seed"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data_cfg = config["data"]
    train_cfg = config["training"]
    data_dir = args.data_dir or data_cfg["era5_monthly_dir"]

    variable_list = [
        "t2m", "tp", "sst", "swvl1", "sd", "siconc",
        "msl", "u10", "v10", "tisr", "ssr", "str",
    ]

    # Compute or load normalization stats
    norm_path = Path(data_cfg["era5_processed_dir"]) / "norm_stats.json"
    if norm_path.exists():
        norm_stats = load_statistics(str(norm_path))
        print(f"Loaded normalization stats from {norm_path}")
    else:
        norm_stats = compute_statistics(
            data_dir, variable_list,
            years=tuple(data_cfg["train_years"]),
        )
        save_statistics(norm_stats, str(norm_path))

    # Create datasets
    train_dataset = WeatherDataset(
        data_dir=data_dir,
        variable_list=variable_list,
        lead_time=data_cfg["lead_times"][0],
        years=tuple(data_cfg["train_years"]),
        norm_stats=norm_stats,
    )

    val_dataset = WeatherDataset(
        data_dir=data_dir,
        variable_list=variable_list,
        lead_time=data_cfg["lead_times"][0],
        years=tuple(data_cfg["val_years"]),
        norm_stats=norm_stats,
    )

    batch_size = args.batch_size or train_cfg["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = setup_model(config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Losses
    losses = setup_losses(config)
    loss_weights = train_cfg.get("loss_weights", {"t2m": 1.0, "tp": 0.5})

    # Optimizer
    lr = args.lr or train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    num_epochs = args.epochs or train_cfg["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - train_cfg.get("warmup_epochs", 5),
        eta_min=lr * 0.01,
    )

    # Mixed precision
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Training loop
    history = {"train": [], "val": []}

    for epoch in range(start_epoch, num_epochs):
        # Warmup
        warmup_epochs = train_cfg.get("warmup_epochs", 5)
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_losses = train_epoch(
            model, train_loader, losses, optimizer, loss_weights,
            device, scaler, use_amp,
        )
        history["train"].append(train_losses)

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validate
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_losses = validate(
                model, val_loader, losses, loss_weights, device,
            )
            history["val"].append({"epoch": epoch, **val_losses})

            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: total={train_losses['total']:.4f} "
                  f"t2m={train_losses['t2m']:.4f} "
                  f"tp={train_losses['tp']:.4f} | "
                  f"Val: total={val_losses['total']:.4f} "
                  f"t2m={val_losses['t2m']:.4f} "
                  f"tp={val_losses['tp']:.4f} "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

            # Save best
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                torch.save(checkpoint, output_dir / "best_model.pt")
                print(f"  → Best model saved (val_loss={best_val_loss:.4f})")
        else:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: total={train_losses['total']:.4f} "
                  f"t2m={train_losses['t2m']:.4f} "
                  f"tp={train_losses['tp']:.4f}")

        # Regular checkpoint
        if epoch % train_cfg.get("save_every", 10) == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")

    # Save final model
    torch.save({
        "epoch": num_epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
    }, output_dir / "final_model.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
