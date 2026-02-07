import torch
import yaml
import argparse
from pathlib import Path
import pytorch_lightning as pl
import json
from datetime import datetime
import warnings

from src.battery_datamodule import BatteryDataModule
from src.lightning_model import BatteryLightningModel

# Suppress warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


def print_test_header(checkpoint_path, args):
    """Print testing information"""
    print("\n" + "=" * 80)
    print("🧪 TESTING BATTERY PREDICTION MODEL")
    print("=" * 80)
    print(f"Checkpoint:        {checkpoint_path}")
    print(f"Test name:         {args.test_name}")
    print("=" * 80 + "\n")


def save_test_results(model, checkpoint_path, output_dir, args):
    """Save comprehensive test results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "test_report"):
        print("⚠️  No test report available from model")
        return

    # Add metadata to report
    report = model.test_report.copy()
    report["test_metadata"] = {
        "checkpoint_path": str(checkpoint_path),
        "test_name": args.test_name,
        "timestamp": datetime.now().isoformat(),
        "batch_size": args.batch_size,
        "devices": args.devices,
    }

    # Save JSON report
    report_path = output_dir / f"{args.test_name}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save latest
    latest_path = output_dir / "test_report_latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Test results saved:")
    print(f"  ├─ Report:     {report_path}")
    print(f"  └─ Latest:     {latest_path}")

    # Print summary to console
    print_test_summary(report)


def print_test_summary(report):
    """Print test summary to console"""
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)

    # Microstructure
    print("\n🔬 MICROSTRUCTURE METRICS:")
    print(f"  Average R²: {report['microstructure_metrics']['average_r2']:.4f}")
    print("  " + "-" * 76)
    print(f"  {'Feature':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("  " + "-" * 76)

    for feature, metrics in report["microstructure_metrics"]["per_feature"].items():
        print(
            f"  {feature:<20} {metrics['MSE']:<12.6f} {metrics['RMSE']:<12.6f} "
            f"{metrics['MAE']:<12.6f} {metrics['R2']:<12.6f}"
        )

    # Performance
    print("\n⚡ PERFORMANCE METRICS:")
    print(f"  Average R²: {report['performance_metrics']['average_r2']:.4f}")
    print("  " + "-" * 76)
    print(f"  {'Feature':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("  " + "-" * 76)

    for feature, metrics in report["performance_metrics"]["per_feature"].items():
        print(
            f"  {feature:<20} {metrics['MSE']:<12.6f} {metrics['RMSE']:<12.6f} "
            f"{metrics['MAE']:<12.6f} {metrics['R2']:<12.6f}"
        )

    print("\n" + "=" * 80)


def main(args):
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load configs
    with open("configs/paths.yml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/train_config.yml", "r") as f:
        train_config = yaml.safe_load(f)

    print_test_header(checkpoint_path, args)

    # Data module
    datamodule = BatteryDataModule(
        data_dir=paths_config["data"]["output"]["optimized_dir"],
        batch_size=train_config["training"]["batch_size"],
        num_workers=train_config["training"]["num_workers"],
    )

    # Load model from checkpoint
    print("Loading model from checkpoint...")
    model = BatteryLightningModel.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config=train_config,
        strict=True,
    )
    print("✓ Model loaded successfully")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nModel parameters: {trainable_params:,} trainable / {total_params:,} total"
    )

    # Setup trainer for testing
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=train_config["training"].get("devices", 1),
        precision=train_config["training"].get("precision", "32"),
        logger=False,  # No logging during testing
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    # Run test
    print("\n" + "=" * 80)
    print("Running test on test dataset...")
    print("=" * 80 + "\n")

    test_results = trainer.test(model, datamodule=datamodule)

    # Save results
    output_dir = Path("test_results") / args.test_name
    save_test_results(model, checkpoint_path, output_dir, args)

    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test trained battery prediction model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., experiments/baseline/version_0/checkpoints/best.ckpt)",
    )

    parser.add_argument(
        "--test_name",
        type=str,
        default="test_run",
        help="Name for this test run (used for output directory)",
    )

    args = parser.parse_args()
    main(args)
