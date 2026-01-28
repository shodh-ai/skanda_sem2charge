import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import pytorch_lightning as pl
from scipy import stats
import pandas as pd

from src.battery_datamodule import BatteryDataModule
from src.lightning_model import BatteryLightningModel


def create_detailed_visualizations(model, output_dir):
    """Create comprehensive visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate all predictions
    all_preds_micro = torch.cat([p["microstructure"] for p in model.test_predictions])
    all_preds_perf = torch.cat([p["performance"] for p in model.test_predictions])
    all_targets_micro = torch.cat([t["microstructure"] for t in model.test_targets])
    all_targets_perf = torch.cat([t["performance"] for t in model.test_targets])

    # Convert to numpy
    preds_micro = all_preds_micro.numpy()
    preds_perf = all_preds_perf.numpy()
    targets_micro = all_targets_micro.numpy()
    targets_perf = all_targets_perf.numpy()

    # Feature names
    micro_names = ["Porosity", "Tortuosity", "Surface Area", "Pore Size Distribution"]
    perf_names = [
        "Capacity",
        "Energy Density",
        "Power Density",
        "Cycle Life",
        "Efficiency",
    ]

    # === 1. Microstructure Scatter Plots ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Microstructure: Predicted vs Ground Truth", fontsize=16, fontweight="bold"
    )

    for idx in range(4):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(
            targets_micro[:, idx],
            preds_micro[:, idx],
            alpha=0.6,
            s=30,
            edgecolors="black",
            linewidth=0.5,
        )

        # Perfect prediction line
        min_val = min(targets_micro[:, idx].min(), preds_micro[:, idx].min())
        max_val = max(targets_micro[:, idx].max(), preds_micro[:, idx].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2.5,
            label="Perfect Prediction",
        )

        # Calculate metrics
        mse = np.mean((preds_micro[:, idx] - targets_micro[:, idx]) ** 2)
        mae = np.mean(np.abs(preds_micro[:, idx] - targets_micro[:, idx]))
        r2 = stats.pearsonr(targets_micro[:, idx], preds_micro[:, idx])[0] ** 2

        # Add metrics text box
        textstr = f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {np.sqrt(mse):.4f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_xlabel("Ground Truth", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_title(micro_names[idx], fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(
        output_dir / "microstructure_predictions.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # === 2. Performance Scatter Plots ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Performance: Predicted vs Ground Truth", fontsize=16, fontweight="bold"
    )

    for idx in range(5):
        ax = axes[idx // 3, idx % 3]
        ax.scatter(
            targets_perf[:, idx],
            preds_perf[:, idx],
            alpha=0.6,
            s=30,
            color="green",
            edgecolors="black",
            linewidth=0.5,
        )

        # Perfect prediction line
        min_val = min(targets_perf[:, idx].min(), preds_perf[:, idx].min())
        max_val = max(targets_perf[:, idx].max(), preds_perf[:, idx].max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2.5,
            label="Perfect Prediction",
        )

        # Calculate metrics
        mse = np.mean((preds_perf[:, idx] - targets_perf[:, idx]) ** 2)
        mae = np.mean(np.abs(preds_perf[:, idx] - targets_perf[:, idx]))
        r2 = stats.pearsonr(targets_perf[:, idx], preds_perf[:, idx])[0] ** 2

        # Add metrics text box
        textstr = f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {np.sqrt(mse):.4f}"
        props = dict(boxstyle="round", facecolor="lightgreen", alpha=0.8)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        ax.set_xlabel("Ground Truth", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_title(perf_names[idx], fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="--")

    # Remove empty subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig(
        output_dir / "performance_predictions.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # === 3. Error Distribution Plots ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Microstructure errors
    errors_micro = np.abs(preds_micro - targets_micro)
    axes[0].hist(
        errors_micro.flatten(),
        bins=60,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=1.2,
    )
    axes[0].axvline(
        errors_micro.mean(),
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Mean Error: {errors_micro.mean():.4f}",
    )
    axes[0].axvline(
        np.median(errors_micro),
        color="green",
        linestyle="-.",
        linewidth=2.5,
        label=f"Median Error: {np.median(errors_micro):.4f}",
    )
    axes[0].set_xlabel("Absolute Error", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Frequency", fontsize=13, fontweight="bold")
    axes[0].set_title(
        "Microstructure Prediction Error Distribution", fontsize=14, fontweight="bold"
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle="--")

    # Performance errors
    errors_perf = np.abs(preds_perf - targets_perf)
    axes[1].hist(
        errors_perf.flatten(),
        bins=60,
        alpha=0.7,
        color="seagreen",
        edgecolor="black",
        linewidth=1.2,
    )
    axes[1].axvline(
        errors_perf.mean(),
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=f"Mean Error: {errors_perf.mean():.4f}",
    )
    axes[1].axvline(
        np.median(errors_perf),
        color="orange",
        linestyle="-.",
        linewidth=2.5,
        label=f"Median Error: {np.median(errors_perf):.4f}",
    )
    axes[1].set_xlabel("Absolute Error", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Frequency", fontsize=13, fontweight="bold")
    axes[1].set_title(
        "Performance Prediction Error Distribution", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "error_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # === 4. Create detailed metrics CSV ===
    metrics_data = []

    for idx, name in enumerate(micro_names):
        mse = np.mean((preds_micro[:, idx] - targets_micro[:, idx]) ** 2)
        mae = np.mean(np.abs(preds_micro[:, idx] - targets_micro[:, idx]))
        rmse = np.sqrt(mse)
        r2 = stats.pearsonr(targets_micro[:, idx], preds_micro[:, idx])[0] ** 2

        metrics_data.append(
            {
                "Output_Type": "Microstructure",
                "Parameter": name,
                "MSE": f"{mse:.6f}",
                "RMSE": f"{rmse:.6f}",
                "MAE": f"{mae:.6f}",
                "R2_Score": f"{r2:.6f}",
                "Mean_Target": f"{targets_micro[:, idx].mean():.4f}",
                "Std_Target": f"{targets_micro[:, idx].std():.4f}",
            }
        )

    for idx, name in enumerate(perf_names):
        mse = np.mean((preds_perf[:, idx] - targets_perf[:, idx]) ** 2)
        mae = np.mean(np.abs(preds_perf[:, idx] - targets_perf[:, idx]))
        rmse = np.sqrt(mse)
        r2 = stats.pearsonr(targets_perf[:, idx], preds_perf[:, idx])[0] ** 2

        metrics_data.append(
            {
                "Output_Type": "Performance",
                "Parameter": name,
                "MSE": f"{mse:.6f}",
                "RMSE": f"{rmse:.6f}",
                "MAE": f"{mae:.6f}",
                "R2_Score": f"{r2:.6f}",
                "Mean_Target": f"{targets_perf[:, idx].mean():.4f}",
                "Std_Target": f"{targets_perf[:, idx].std():.4f}",
            }
        )

    df = pd.DataFrame(metrics_data)
    df.to_csv(output_dir / "detailed_metrics_per_parameter.csv", index=False)

    # Print summary
    print(f"\n✅ Visualizations created:")
    print(f"  ├─ {output_dir / 'microstructure_predictions.png'}")
    print(f"  ├─ {output_dir / 'performance_predictions.png'}")
    print(f"  ├─ {output_dir / 'error_distributions.png'}")
    print(f"  └─ {output_dir / 'detailed_metrics_per_parameter.csv'}")

    print("\n" + "=" * 80)
    print("📋 DETAILED METRICS PER PARAMETER")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test trained battery prediction model"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--experiment", type=str, help="Experiment name to test")
    args = parser.parse_args()

    # Load configs
    with open("configs/paths.yml", "r") as f:
        paths_config = yaml.safe_load(f)

    with open("configs/train_config.yml", "r") as f:
        train_config = yaml.safe_load(f)

    print("=" * 80)
    print("🧪 Testing Battery Prediction Model")
    print("=" * 80)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.experiment:
        exp_dir = Path("experiments") / args.experiment / "checkpoints"
        checkpoints = list(exp_dir.glob("*.ckpt"))
        if not checkpoints:
            print(f"❌ No checkpoints found in {exp_dir}")
            return
        # Get best checkpoint (lowest val_loss in filename)
        checkpoint_path = min(
            checkpoints, key=lambda p: float(p.stem.split("val_loss")[-1])
        )
        print(f"Using best checkpoint: {checkpoint_path}")
    else:
        # Find latest experiment
        exp_dirs = sorted(
            Path("experiments").glob("*/checkpoints"), key=lambda p: p.stat().st_mtime
        )
        if not exp_dirs:
            print("❌ No experiments found!")
            return
        checkpoints = list(exp_dirs[-1].glob("*.ckpt"))
        if not checkpoints:
            print(f"❌ No checkpoints found!")
            return
        checkpoint_path = min(
            checkpoints, key=lambda p: float(p.stem.split("val_loss")[-1])
        )
        print(f"Using checkpoint: {checkpoint_path}")

    # Setup datamodule
    datamodule = BatteryDataModule(
        data_dir=paths_config["data"]["output"]["optimized_dir"],
        batch_size=train_config["training"]["batch_size"],
        num_workers=train_config["training"]["num_workers"],
    )

    # Load model
    model = BatteryLightningModel.load_from_checkpoint(
        checkpoint_path, config=train_config
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
    )

    # Run test
    print("\nRunning comprehensive testing...\n")
    trainer.test(model, datamodule)

    # Create visualizations
    output_dir = Path(checkpoint_path).parent.parent / "test_results"
    create_detailed_visualizations(model, output_dir)

    # Save updated report with visualizations
    if hasattr(model, "test_report"):
        report = model.test_report.copy()
        report["metadata"]["visualization_dir"] = str(output_dir)
        report["metadata"]["test_timestamp"] = datetime.now().isoformat()

        report_path = output_dir / "comprehensive_test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Comprehensive report saved: {report_path}\n")


if __name__ == "__main__":
    main()
