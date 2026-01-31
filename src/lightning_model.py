import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    MeanAbsolutePercentageError,
)
from .models.simple_model import SimpleModel
import json
from pathlib import Path
from typing import Dict
import numpy as np

# Import shared feature config
from utils.feature_config import (
    MICROSTRUCTURE_SHORT,
    PERFORMANCE_SHORT,
    get_loss_weight_tensors,
    NUM_INPUT_FEATURES,
    NUM_MICROSTRUCTURE_OUTPUTS,
    NUM_PERFORMANCE_OUTPUTS,
)


class BatteryLightningModel(pl.LightningModule):
    """
    Lightning wrapper for battery prediction with comprehensive per-feature metrics.

    Features:
    - Separate R² tracking for each output feature
    - Weighted multi-task loss
    - Detailed logging and reporting
    - Support for feature-specific analysis
    """

    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        perf_weights, micro_weights = get_loss_weight_tensors()
        self.performance_weights = perf_weights
        self.microstructure_weights = micro_weights

        # Model
        self.model = SimpleModel(
            input_params_dim=NUM_INPUT_FEATURES,
            micro_output_dim=NUM_MICROSTRUCTURE_OUTPUTS,
            perf_output_dim=NUM_PERFORMANCE_OUTPUTS,
        )

        # Base loss functions
        self.mse = nn.MSELoss(reduction="none")
        self.mae = nn.L1Loss(reduction="none")

        # ========== PER-FEATURE METRICS ==========

        # Training metrics - per feature R²
        self.train_micro_r2_per_feature = nn.ModuleList([R2Score() for _ in range(4)])
        self.train_perf_r2_per_feature = nn.ModuleList([R2Score() for _ in range(5)])

        # Validation metrics - per feature
        self.val_micro_r2_per_feature = nn.ModuleList([R2Score() for _ in range(4)])
        self.val_perf_r2_per_feature = nn.ModuleList([R2Score() for _ in range(5)])

        # Test metrics - comprehensive per feature
        self.test_micro_metrics = nn.ModuleDict(
            {
                "mse": nn.ModuleList([MeanSquaredError() for _ in range(4)]),
                "mae": nn.ModuleList([MeanAbsoluteError() for _ in range(4)]),
                "r2": nn.ModuleList([R2Score() for _ in range(4)]),
                "mape": nn.ModuleList(
                    [MeanAbsolutePercentageError() for _ in range(4)]
                ),
            }
        )

        self.test_perf_metrics = nn.ModuleDict(
            {
                "mse": nn.ModuleList([MeanSquaredError() for _ in range(5)]),
                "mae": nn.ModuleList([MeanAbsoluteError() for _ in range(5)]),
                "r2": nn.ModuleList([R2Score() for _ in range(5)]),
                "mape": nn.ModuleList(
                    [MeanAbsolutePercentageError() for _ in range(5)]
                ),
            }
        )

        # Store test predictions for detailed analysis
        self.test_predictions = []
        self.test_targets = []
        self.test_metadata = []

    def forward(self, image, params):
        """Forward pass through model"""
        return self.model(image, params)

    def compute_weighted_loss(
        self,
        pred_micro: torch.Tensor,
        target_micro: torch.Tensor,
        pred_perf: torch.Tensor,
        target_perf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss with per-feature weighting.

        Args:
            pred_micro: [B, 4] microstructure predictions
            target_micro: [B, 4] microstructure targets
            pred_perf: [B, 5] performance predictions
            target_perf: [B, 5] performance targets

        Returns:
            Dictionary with total_loss, micro_loss, perf_loss, and per-feature losses
        """
        # Per-feature MSE: [B, 4] and [B, 5]
        micro_mse_per_feat = self.mse(pred_micro, target_micro)  # [B, 4]
        perf_mse_per_feat = self.mse(pred_perf, target_perf)  # [B, 5]

        # Apply weights and reduce: mean over batch, weighted sum over features
        micro_weights = self.microstructure_weights.to(pred_micro.device)
        perf_weights = self.performance_weights.to(pred_perf.device)

        # Weighted loss per feature, then sum
        micro_loss = (micro_mse_per_feat * micro_weights).sum(dim=1).mean()
        perf_loss = (perf_mse_per_feat * perf_weights).sum(dim=1).mean()

        # Total loss
        total_loss = micro_loss + perf_loss

        # Store individual feature losses for logging
        micro_losses_dict = {
            f"micro_{MICROSTRUCTURE_SHORT[i]}_loss": micro_mse_per_feat[:, i].mean()
            for i in range(4)
        }
        perf_losses_dict = {
            f"perf_{PERFORMANCE_SHORT[i]}_loss": perf_mse_per_feat[:, i].mean()
            for i in range(5)
        }

        return {
            "total_loss": total_loss,
            "micro_loss": micro_loss,
            "perf_loss": perf_loss,
            **micro_losses_dict,
            **perf_losses_dict,
        }

    def training_step(self, batch, batch_idx):
        """Training step with per-feature metric updates"""
        pred = self(batch["image"], batch["input_params"])
        target_micro = batch["microstructure_outputs"]
        target_perf = batch["performance_outputs"]

        # Compute losses
        losses = self.compute_weighted_loss(
            pred["microstructure"], target_micro, pred["performance"], target_perf
        )

        # Update per-feature R² metrics
        for i in range(4):
            self.train_micro_r2_per_feature[i](
                pred["microstructure"][:, i], target_micro[:, i]
            )

        for i in range(5):
            self.train_perf_r2_per_feature[i](
                pred["performance"][:, i], target_perf[:, i]
            )

        # Log main losses
        self.log(
            "train_loss",
            losses["total_loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_micro_loss", losses["micro_loss"], on_epoch=True, sync_dist=True
        )
        self.log("train_perf_loss", losses["perf_loss"], on_epoch=True, sync_dist=True)

        # Log per-feature losses (only on epoch end to reduce clutter)
        for key, value in losses.items():
            if key not in ["total_loss", "micro_loss", "perf_loss"]:
                self.log(f"train_{key}", value, on_epoch=True, sync_dist=True)

        return losses["total_loss"]

    def on_train_epoch_end(self):
        """Log per-feature R² scores at epoch end"""
        # Microstructure R²
        for i, name in enumerate(MICROSTRUCTURE_SHORT):
            r2 = self.train_micro_r2_per_feature[i].compute()
            self.log(f"train_micro_r2_{name}", r2, sync_dist=True)
            self.train_micro_r2_per_feature[i].reset()

        # Performance R²
        for i, name in enumerate(PERFORMANCE_SHORT):
            r2 = self.train_perf_r2_per_feature[i].compute()
            self.log(f"train_perf_r2_{name}", r2, sync_dist=True)
            self.train_perf_r2_per_feature[i].reset()

        # Compute average R² for overall tracking
        micro_r2_avg = torch.stack(
            [self.train_micro_r2_per_feature[i].compute() for i in range(4)]
        ).mean()
        perf_r2_avg = torch.stack(
            [self.train_perf_r2_per_feature[i].compute() for i in range(5)]
        ).mean()

        self.log("train_micro_r2_avg", micro_r2_avg, sync_dist=True)
        self.log("train_perf_r2_avg", perf_r2_avg, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Validation step with per-feature metrics"""
        pred = self(batch["image"], batch["input_params"])
        target_micro = batch["microstructure_outputs"]
        target_perf = batch["performance_outputs"]

        # Compute losses
        losses = self.compute_weighted_loss(
            pred["microstructure"], target_micro, pred["performance"], target_perf
        )

        # Update per-feature R² metrics
        for i in range(4):
            self.val_micro_r2_per_feature[i](
                pred["microstructure"][:, i], target_micro[:, i]
            )

        for i in range(5):
            self.val_perf_r2_per_feature[i](
                pred["performance"][:, i], target_perf[:, i]
            )

        # Log main losses
        self.log("val_loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        self.log("val_micro_loss", losses["micro_loss"], sync_dist=True)
        self.log("val_perf_loss", losses["perf_loss"], sync_dist=True)

        # Log per-feature losses
        for key, value in losses.items():
            if key not in ["total_loss", "micro_loss", "perf_loss"]:
                self.log(f"val_{key}", value, sync_dist=True)

        return losses["total_loss"]

    def on_validation_epoch_end(self):
        """Log per-feature R² scores at validation epoch end"""
        # Microstructure R²
        for i, name in enumerate(MICROSTRUCTURE_SHORT):
            r2 = self.val_micro_r2_per_feature[i].compute()
            self.log(
                f"val_micro_r2_{name}", r2, prog_bar=(i == 2), sync_dist=True
            )  # Show tau in progress bar
            self.val_micro_r2_per_feature[i].reset()

        # Performance R²
        for i, name in enumerate(PERFORMANCE_SHORT):
            r2 = self.val_perf_r2_per_feature[i].compute()
            self.log(
                f"val_perf_r2_{name}", r2, prog_bar=(i == 0), sync_dist=True
            )  # Show cycle_life in progress bar
            self.val_perf_r2_per_feature[i].reset()

        # Average R² for overall tracking
        micro_r2_avg = torch.stack(
            [self.val_micro_r2_per_feature[i].compute() for i in range(4)]
        ).mean()
        perf_r2_avg = torch.stack(
            [self.val_perf_r2_per_feature[i].compute() for i in range(5)]
        ).mean()

        self.log("val_micro_r2_avg", micro_r2_avg, sync_dist=True)
        self.log("val_perf_r2_avg", perf_r2_avg, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """Test step with comprehensive per-feature metrics"""
        pred = self(batch["image"], batch["input_params"])
        target_micro = batch["microstructure_outputs"]
        target_perf = batch["performance_outputs"]

        # Compute losses
        losses = self.compute_weighted_loss(
            pred["microstructure"], target_micro, pred["performance"], target_perf
        )

        # Update ALL per-feature metrics
        for i in range(4):
            pred_i = pred["microstructure"][:, i]
            target_i = target_micro[:, i]

            self.test_micro_metrics["mse"][i](pred_i, target_i)
            self.test_micro_metrics["mae"][i](pred_i, target_i)
            self.test_micro_metrics["r2"][i](pred_i, target_i)
            self.test_micro_metrics["mape"][i](pred_i, target_i)

        for i in range(5):
            pred_i = pred["performance"][:, i]
            target_i = target_perf[:, i]

            self.test_perf_metrics["mse"][i](pred_i, target_i)
            self.test_perf_metrics["mae"][i](pred_i, target_i)
            self.test_perf_metrics["r2"][i](pred_i, target_i)
            self.test_perf_metrics["mape"][i](pred_i, target_i)

        # Store predictions for detailed analysis
        self.test_predictions.append(
            {
                "microstructure": pred["microstructure"].detach().cpu(),
                "performance": pred["performance"].detach().cpu(),
            }
        )
        self.test_targets.append(
            {
                "microstructure": target_micro.detach().cpu(),
                "performance": target_perf.detach().cpu(),
            }
        )
        self.test_metadata.append(
            {
                "sample_ids": batch.get("sample_ids", []),
                "param_ids": batch.get("param_ids", []),
            }
        )

        # Log basic losses
        self.log("test_loss", losses["total_loss"])
        self.log("test_micro_loss", losses["micro_loss"])
        self.log("test_perf_loss", losses["perf_loss"])

        return losses["total_loss"]

    def on_test_epoch_end(self):
        """Compute comprehensive per-feature test metrics and create report"""

        # ========== MICROSTRUCTURE METRICS ==========
        micro_results = {}
        for i, name in enumerate(MICROSTRUCTURE_SHORT):
            mse_val = self.test_micro_metrics["mse"][i].compute()
            mae_val = self.test_micro_metrics["mae"][i].compute()
            r2_val = self.test_micro_metrics["r2"][i].compute()
            mape_val = self.test_micro_metrics["mape"][i].compute()
            rmse_val = torch.sqrt(mse_val)

            micro_results[name] = {
                "MSE": float(mse_val),
                "RMSE": float(rmse_val),
                "MAE": float(mae_val),
                "R2": float(r2_val),
                "MAPE": float(mape_val),
            }

            # Log individual metrics
            self.log(f"test_micro_mse_{name}", mse_val)
            self.log(f"test_micro_rmse_{name}", rmse_val)
            self.log(f"test_micro_mae_{name}", mae_val)
            self.log(f"test_micro_r2_{name}", r2_val)
            self.log(f"test_micro_mape_{name}", mape_val)

        # ========== PERFORMANCE METRICS ==========
        perf_results = {}
        for i, name in enumerate(PERFORMANCE_SHORT):
            mse_val = self.test_perf_metrics["mse"][i].compute()
            mae_val = self.test_perf_metrics["mae"][i].compute()
            r2_val = self.test_perf_metrics["r2"][i].compute()
            mape_val = self.test_perf_metrics["mape"][i].compute()
            rmse_val = torch.sqrt(mse_val)

            perf_results[name] = {
                "MSE": float(mse_val),
                "RMSE": float(rmse_val),
                "MAE": float(mae_val),
                "R2": float(r2_val),
                "MAPE": float(mape_val),
            }

            # Log individual metrics
            self.log(f"test_perf_mse_{name}", mse_val)
            self.log(f"test_perf_rmse_{name}", rmse_val)
            self.log(f"test_perf_mae_{name}", mae_val)
            self.log(f"test_perf_r2_{name}", r2_val)
            self.log(f"test_perf_mape_{name}", mape_val)

        # ========== AVERAGE METRICS ==========
        avg_micro_r2 = np.mean(
            [micro_results[name]["R2"] for name in MICROSTRUCTURE_SHORT]
        )
        avg_perf_r2 = np.mean([perf_results[name]["R2"] for name in PERFORMANCE_SHORT])

        self.log("test_micro_r2_avg", avg_micro_r2)
        self.log("test_perf_r2_avg", avg_perf_r2)

        # ========== CREATE COMPREHENSIVE REPORT ==========
        report = {
            "model_info": {
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
                "loss_weights": {
                    "microstructure": self.microstructure_weights.tolist(),
                    "performance": self.performance_weights.tolist(),
                },
            },
            "microstructure_metrics": {
                "per_feature": micro_results,
                "average_r2": avg_micro_r2,
            },
            "performance_metrics": {
                "per_feature": perf_results,
                "average_r2": avg_perf_r2,
            },
        }

        # Store report
        self.test_report = report

        # Print formatted report
        self._print_detailed_test_report(report)

        # Save detailed predictions
        self._save_predictions()

    def _print_detailed_test_report(self, report):
        """Print beautifully formatted test report to console"""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE TEST RESULTS - PER-FEATURE BREAKDOWN")
        print("=" * 100)

        # Model info
        print("\n🏗️  MODEL INFORMATION:")
        print(
            f"  ├─ Total Parameters:      {report['model_info']['total_parameters']:,}"
        )
        print(
            f"  └─ Trainable Parameters:  {report['model_info']['trainable_parameters']:,}"
        )

        # Microstructure results
        print("\n🔬 MICROSTRUCTURE PREDICTIONS (4 features):")
        print(f"  Average R²: {report['microstructure_metrics']['average_r2']:.4f}")
        print("  " + "-" * 96)
        print(
            f"  {'Feature':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'MAPE':<12}"
        )
        print("  " + "-" * 96)

        for name in MICROSTRUCTURE_SHORT:
            m = report["microstructure_metrics"]["per_feature"][name]
            print(
                f"  {name:<20} {m['MSE']:<12.6f} {m['RMSE']:<12.6f} {m['MAE']:<12.6f} {m['R2']:<12.6f} {m['MAPE']:<12.4f}%"
            )

        # Performance results
        print("\n⚡ PERFORMANCE PREDICTIONS (5 features):")
        print(f"  Average R²: {report['performance_metrics']['average_r2']:.4f}")
        print("  " + "-" * 96)
        print(
            f"  {'Feature':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'MAPE':<12}"
        )
        print("  " + "-" * 96)

        for name in PERFORMANCE_SHORT:
            p = report["performance_metrics"]["per_feature"][name]
            print(
                f"  {name:<20} {p['MSE']:<12.6f} {p['RMSE']:<12.6f} {p['MAE']:<12.6f} {p['R2']:<12.6f} {p['MAPE']:<12.4f}%"
            )

        print("\n" + "=" * 100)

        # Highlight best and worst performing features
        print("\n🎯 FEATURE PERFORMANCE HIGHLIGHTS:")

        all_r2s = {
            **{
                f"micro_{k}": v["R2"]
                for k, v in report["microstructure_metrics"]["per_feature"].items()
            },
            **{
                f"perf_{k}": v["R2"]
                for k, v in report["performance_metrics"]["per_feature"].items()
            },
        }

        best_feature = max(all_r2s.items(), key=lambda x: x[1])
        worst_feature = min(all_r2s.items(), key=lambda x: x[1])

        print(f"  🏆 Best:  {best_feature[0]:<25} R² = {best_feature[1]:.4f}")
        print(f"  ⚠️  Worst: {worst_feature[0]:<25} R² = {worst_feature[1]:.4f}")

        print("=" * 100 + "\n")

    def _save_predictions(self):
        """Save detailed predictions to file"""
        if len(self.test_predictions) == 0:
            return

        save_dir = Path(self.logger.log_dir) / "test_results"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Concatenate all predictions
        all_pred_micro = torch.cat([p["microstructure"] for p in self.test_predictions])
        all_pred_perf = torch.cat([p["performance"] for p in self.test_predictions])
        all_target_micro = torch.cat([t["microstructure"] for t in self.test_targets])
        all_target_perf = torch.cat([t["performance"] for t in self.test_targets])

        # Save as numpy arrays
        np.save(save_dir / "predictions_microstructure.npy", all_pred_micro.numpy())
        np.save(save_dir / "predictions_performance.npy", all_pred_perf.numpy())
        np.save(save_dir / "targets_microstructure.npy", all_target_micro.numpy())
        np.save(save_dir / "targets_performance.npy", all_target_perf.numpy())

        # Save report as JSON
        with open(save_dir / "test_report.json", "w") as f:
            json.dump(self.test_report, f, indent=2)

        print(f"💾 Test results saved to: {save_dir}")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        opt_cfg = self.config["optimizer"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=opt_cfg.get("betas", (0.9, 0.999)),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config["scheduler"]["factor"],
            patience=self.config["scheduler"]["patience"],
            min_lr=self.config["scheduler"]["min_lr"],
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
