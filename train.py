import torch
import yaml
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from src.battery_datamodule import BatteryDataModule
from src.lightning_model import BatteryLightningModel


@rank_zero_only
def print_config(args, train_config, paths_config, total_params):
    print("=" * 80)
    print("🔋 Training Battery Prediction Model")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Epochs: {train_config['training']['epochs']}")
    print(f"Batch size (per GPU): {train_config['training']['batch_size']}")
    print(
        f"Total Effective Batch size: {train_config['training']['batch_size'] * train_config['training'].get('devices', 1) if train_config['training'].get('devices') != -1 else 'Dynamic'}"
    )
    print(f"Data dir: {paths_config['data']['output']['optimized_dir']}")
    print(f"Total parameters: {total_params:,}")
    print("=" * 80)


@rank_zero_only
def print_completion(checkpoint_path, log_dir):
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Logs: {log_dir}")
    print("=" * 80)


def main(args):
    # Seed
    pl.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("medium")

    # Load configs
    with open("configs/paths.yml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/train_config.yml", "r") as f:
        train_config = yaml.safe_load(f)

    # Data
    datamodule = BatteryDataModule(
        data_dir=paths_config["data"]["output"]["optimized_dir"],
        batch_size=train_config["training"]["batch_size"],
        num_workers=train_config["training"]["num_workers"],
    )

    # Model
    model = BatteryLightningModel(train_config)

    # Count params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_config(args, train_config, paths_config, total_params)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("experiments") / args.experiment_name / "checkpoints",
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=train_config["training"]["early_stopping"]["patience"],
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Logger
    logger = TensorBoardLogger(save_dir="experiments", name=args.experiment_name)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config["training"]["epochs"],
        accelerator=train_config["training"]["accelerator"],
        devices=train_config["training"]["devices"],
        precision=train_config["training"]["precision"],
        gradient_clip_val=train_config["training"]["gradient_clip_val"],
        strategy="ddp",
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        deterministic="warn",
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, datamodule)

    # Test
    trainer.test(model, datamodule, ckpt_path="best")

    print_completion(checkpoint_callback.best_model_path, logger.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, default="baseline", help="Experiment name"
    )
    args = parser.parse_args()
    main(args)
