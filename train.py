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
from src.utils import get_loss_weight_lists


@rank_zero_only
def print_training_info(args, train_config, paths_config, total_params):
    """Print training configuration"""
    perf_weights, micro_weights = get_loss_weight_lists()
    devices = train_config["training"].get("devices", 1)
    batch_size = train_config["training"]["batch_size"]
    effective_batch = "Dynamic (all GPUs)" if devices == -1 else batch_size * devices

    print("\n" + "=" * 80)
    print("🔋 BATTERY PREDICTION MODEL TRAINING")
    print("=" * 80)
    print(f"Experiment:        {args.experiment_name}")
    print(f"Epochs:            {train_config['training']['epochs']}")
    print(f"Batch size:        {batch_size} per GPU")
    print(f"Effective batch:   {effective_batch}")
    print(f"Parameters:        {total_params:,}")
    print(f"Model size:        {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    print(f"Data dir:          {paths_config['data']['output']['optimized_dir']}")
    print(f"\n📊 Loss Weights:")
    print(f"  Performance:     {perf_weights}")
    print(f"  Microstructure:  {micro_weights}")
    print("=" * 80 + "\n")


@rank_zero_only
def print_completion(checkpoint_path, log_dir):
    """Print training completion summary"""
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"📁 Experiment dir:  {log_dir}")
    print(f"🏆 Best checkpoint: {checkpoint_path}")
    print(f"📊 TensorBoard:     tensorboard --logdir={log_dir}")
    print("=" * 80 + "\n")


def main(args):
    # Reproducibility
    pl.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("medium")

    # Load configs
    with open("configs/paths.yml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/train_config.yml", "r") as f:
        train_config = yaml.safe_load(f)

    # Data module
    datamodule = BatteryDataModule(
        data_dir=paths_config["data"]["output"]["optimized_dir"],
        batch_size=train_config["training"]["batch_size"],
        num_workers=train_config["training"]["num_workers"],
    )

    # Model
    model = BatteryLightningModel(train_config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print info
    print_training_info(args, train_config, paths_config, total_params)

    # Logger - creates experiments/{experiment_name}/version_X/
    logger = TensorBoardLogger(
        save_dir="experiments",
        name=args.experiment_name,
        default_hp_metric=False,
    )

    # Checkpoint callback - saves to version_X/checkpoints/
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / "checkpoints",
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=False,
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=train_config["training"]["early_stopping"]["patience"],
        mode="min",
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config["training"]["epochs"],
        accelerator=train_config["training"]["accelerator"],
        devices=train_config["training"]["devices"],
        precision=train_config["training"]["precision"],
        gradient_clip_val=train_config["training"]["gradient_clip_val"],
        strategy="ddp" if train_config["training"]["devices"] != 1 else "auto",
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        deterministic="warn",
        log_every_n_steps=10,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
    )

    # Train
    trainer.fit(model, datamodule)

    # Test with best checkpoint
    print("\n" + "=" * 80)
    print("🧪 TESTING BEST MODEL")
    print("=" * 80 + "\n")

    trainer.test(model, datamodule, ckpt_path="best")

    # Print completion
    print_completion(checkpoint_callback.best_model_path, logger.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train battery prediction model")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline",
        help="Experiment name (creates experiments/{name}/version_X/)",
    )
    args = parser.parse_args()
    main(args)
