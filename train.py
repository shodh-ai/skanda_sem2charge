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
import json
from datetime import datetime
import warnings

from src.battery_datamodule import BatteryDataModule
from src.lightning_model import BatteryLightningModel
from src.utils import get_loss_weight_lists


warnings.filterwarnings("ignore", message=".*isinstance.*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*IterableDataset.*__len__.*")
warnings.filterwarnings("ignore", message=".*infer.*batch_size.*")
warnings.filterwarnings("ignore", message=".*compute.*method.*called before.*update.*")
warnings.filterwarnings("ignore", message=".*Precision.*not supported.*model summary.*")
warnings.filterwarnings(
    "ignore", message=".*ModelCheckpoint.*could not find.*monitored key.*"
)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


@rank_zero_only
def print_config(args, train_config, paths_config, total_params):
    perf_weights, micro_weights = get_loss_weight_lists()

    print("=" * 80)
    print("🔋 Training Battery Prediction Model")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Epochs: {train_config['training']['epochs']}")
    print(f"Batch size (per GPU): {train_config['training']['batch_size']}")
    devices = train_config["training"].get("devices", 1)
    if devices == -1:
        print(f"Total Effective Batch size: Dynamic (all GPUs)")
    else:
        print(
            f"Total Effective Batch size: {train_config['training']['batch_size'] * devices}"
        )
    print(f"Data dir: {paths_config['data']['output']['optimized_dir']}")
    print(f"Total parameters: {total_params:,}")
    print(f"\n📊 Loss Weights (from utils/feature_config.py):")
    print(f"  Performance: {perf_weights}")
    print(f"  Microstructure: {micro_weights}")
    print("=" * 80)


@rank_zero_only
def save_test_report(model, experiment_dir, logger):
    """Save comprehensive test report to JSON"""
    if not hasattr(model, "test_report"):
        print("⚠️  No test report available")
        return

    # Add metadata
    report = model.test_report.copy()
    report["metadata"] = {
        "experiment_name": experiment_dir.name,
        "timestamp": datetime.now().isoformat(),
        "tensorboard_log_dir": str(logger.log_dir),
    }

    # Save to experiment directory
    report_dir = experiment_dir / "test_results"
    report_dir.mkdir(exist_ok=True)

    # Save with version number
    version_num = logger.version
    report_path = report_dir / f"test_report_version_{version_num}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Also save as latest
    latest_path = report_dir / "test_report_latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Test report saved:")
    print(f"  ├─ {report_path}")
    print(f"  └─ {latest_path}\n")


@rank_zero_only
def print_completion(checkpoint_path, log_dir, report_path):
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Logs: {log_dir}")
    if report_path:
        print(f"Test report: {report_path}")
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

    # Setup experiment directory
    experiment_dir = Path("experiments") / args.experiment_name

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir / "checkpoints",
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=train_config["training"]["checkpoint"].get("save_last", True),
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=train_config["training"]["early_stopping"]["patience"],
        mode="min",
        verbose=True,
        check_on_train_epoch_end=False,
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
        strategy="ddp" if train_config["training"]["devices"] != 1 else "auto",
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        deterministic="warn",
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
    )

    # Train
    trainer.fit(model, datamodule)

    # Test with best model
    print("\n" + "=" * 80)
    print("🧪 Running comprehensive testing on best model...")
    print("=" * 80)

    trainer.test(model, datamodule, ckpt_path="best")

    # Save test report
    save_test_report(model, experiment_dir, logger)

    # Print completion
    report_path = (
        experiment_dir / "test_results" / f"test_report_version_{logger.version}.json"
    )
    print_completion(checkpoint_callback.best_model_path, logger.log_dir, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, default="baseline", help="Experiment name"
    )
    args = parser.parse_args()
    main(args)
