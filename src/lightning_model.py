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

from .utils import (
    MICROSTRUCTURE_SHORT,
    PERFORMANCE_SHORT,
    get_loss_weight_tensors,
    NUM_INPUT_FEATURES,
    NUM_MICROSTRUCTURE_OUTPUTS,
    NUM_PERFORMANCE_OUTPUTS,
)


class BatteryLightningModel(pl.LightningModule):

    def __init__(self, config: Dict):
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

        self.test_pred_micro = []
        self.test_pred_perf = []
        self.test_target_micro = []
        self.test_target_perf = []

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
        """
        # Per-feature MSE: [B, 4] and [B, 5]
        micro_mse_per_feat = self.mse(pred_micro, target_micro)  # [B, 4]
        perf_mse_per_feat = self.mse(pred_perf, target_perf)  # [B, 5]

        # Apply weights and reduce
        micro_weights = self.microstructure_weights.to(pred_micro.device)
        perf_weights = self.performance_weights.to(pred_perf.device)

        # Weighted loss per feature, then sum
        micro_loss = (micro_mse_per_feat * micro_weights).sum(dim=1).mean()
        perf_loss = (perf_mse_per_feat * perf_weights).sum(dim=1).mean()

        # Total loss
        total_loss = micro_loss + perf_loss

        return {
            "total_loss": total_loss,
            "micro_loss": micro_loss,
            "perf_loss": perf_loss,
        }

    def training_step(self, batch, batch_idx):
        """Training step with per-feature metric updates"""
        pred = self(batch["image"], batch["input_params"])
        pred_micro = pred["microstructure"]
        pred_perf = pred["performance"]
        target_micro = batch["microstructure_outputs"]
        target_perf = batch["performance_outputs"]
        batch_size = pred_micro.shape[0]

        # Compute losses
        losses = self.compute_weighted_loss(
            pred_micro, target_micro, pred_perf, target_perf
        )
        total_loss = losses["total_loss"]
        micro_loss = losses["micro_loss"]
        perf_loss = losses["perf_loss"]

        # Logging in training step
        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_micro_loss",
            micro_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_perf_loss",
            perf_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step - loss + R² metrics"""
        pred = self(batch["image"], batch["input_params"])
        pred_micro = pred["microstructure"]
        pred_perf = pred["performance"]
        target_micro = batch["microstructure_outputs"]
        target_perf = batch["performance_outputs"]

        # Compute losses
        losses = self.compute_weighted_loss(
            pred_micro, target_micro, pred_perf, target_perf
        )
        loss = losses["total_loss"]
        micro_loss = losses["micro_loss"]
        perf_loss = losses["perf_loss"]

        # Logging in validation step
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_micro_loss", micro_loss, sync_dist=True)
        self.log("val_perf_loss", perf_loss, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step - accumulate predictions and targets"""
        pred = self(batch["image"], batch["input_params"])
        pred_micro = pred["microstructure"]
        pred_perf = pred["performance"]
        target_micro = batch["microstructure_outputs"]
        target_perf = batch["performance_outputs"]
        batch_size = pred_micro.shape[0]

        # Compute losses
        losses = self.compute_weighted_loss(
            pred_micro, target_micro, pred_perf, target_perf
        )
        total_loss = losses["total_loss"]
        micro_loss = losses["micro_loss"]
        perf_loss = losses["perf_loss"]

        # Store predictions and targets (detach and move to CPU to save memory)
        self.test_pred_micro.append(pred_micro.detach().cpu())
        self.test_pred_perf.append(pred_perf.detach().cpu())
        self.test_target_micro.append(target_micro.detach().cpu())
        self.test_target_perf.append(target_perf.detach().cpu())

        # Log only losses to tensorboard
        self.log(
            "test_loss",
            total_loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test_micro_loss",
            micro_loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test_perf_loss",
            perf_loss,
            sync_dist=True,
            batch_size=batch_size,
        )

        return total_loss

    def configure_optimizers(self):
        """Configure optimizer with cosine annealing + warmup for large dataset"""
        opt_cfg = self.config["optimizer"]
        sched_cfg = self.config["scheduler"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=opt_cfg.get("betas", (0.9, 0.999)),
            eps=opt_cfg.get("eps", 1e-8),
        )

        # Use cosine annealing with warmup for large datasets
        if sched_cfg["name"] == "cosine_with_warmup":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            warmup_epochs = sched_cfg.get("warmup_epochs", 5)
            total_epochs = self.config["training"]["epochs"]

            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,  # Start from 1% of base LR
                end_factor=1.0,
                total_iters=warmup_epochs,
            )

            # Cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=sched_cfg.get("min_lr", 1e-6),
            )

            # Combine schedulers
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        else:
            # Fallback to ReduceLROnPlateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 10),
                min_lr=sched_cfg.get("min_lr", 1e-6),
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

    def on_test_epoch_end(self):
        """Compute comprehensive metrics from stored predictions"""

        # Concatenate all batches
        pred_micro = torch.cat(self.test_pred_micro, dim=0)  # [N, 4]
        pred_perf = torch.cat(self.test_pred_perf, dim=0)  # [N, 5]
        target_micro = torch.cat(self.test_target_micro, dim=0)  # [N, 4]
        target_perf = torch.cat(self.test_target_perf, dim=0)  # [N, 5]

        # ========== COMPUTE MICROSTRUCTURE METRICS ==========
        micro_results = {}
        for i, name in enumerate(MICROSTRUCTURE_SHORT):
            pred_i = pred_micro[:, i]
            target_i = target_micro[:, i]

            # Instantiate metrics (no state, one-time use)
            mse_metric = MeanSquaredError()
            mae_metric = MeanAbsoluteError()
            r2_metric = R2Score()
            mape_metric = MeanAbsolutePercentageError()

            # Compute on full dataset
            mse = mse_metric(pred_i, target_i).item()
            mae = mae_metric(pred_i, target_i).item()
            r2 = r2_metric(pred_i, target_i).item()
            mape = mape_metric(pred_i, target_i).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()

            micro_results[name] = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "MAPE": mape,
            }

        # ========== COMPUTE PERFORMANCE METRICS ==========
        perf_results = {}
        for i, name in enumerate(PERFORMANCE_SHORT):
            pred_i = pred_perf[:, i]
            target_i = target_perf[:, i]

            # Instantiate metrics (no state, one-time use)
            mse_metric = MeanSquaredError()
            mae_metric = MeanAbsoluteError()
            r2_metric = R2Score()
            mape_metric = MeanAbsolutePercentageError()

            # Compute on full dataset
            mse = mse_metric(pred_i, target_i).item()
            mae = mae_metric(pred_i, target_i).item()
            r2 = r2_metric(pred_i, target_i).item()
            mape = mape_metric(pred_i, target_i).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()

            perf_results[name] = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "MAPE": mape,
            }

        # ========== AVERAGE R² ==========
        avg_micro_r2 = np.mean([v["R2"] for v in micro_results.values()])
        avg_perf_r2 = np.mean([v["R2"] for v in perf_results.values()])

        # ========== CREATE REPORT ==========
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

        # Print to console (not tensorboard)
        self._print_detailed_test_report(report)

        # Save predictions and targets
        self._save_predictions(pred_micro, pred_perf, target_micro, target_perf, report)

    def _save_predictions(
        self, pred_micro, pred_perf, target_micro, target_perf, report
    ):
        """Save predictions, targets, and report to experiment directory"""
        # Save to logger's directory (experiments/{name}/version_X/test_results/)
        if self.trainer.log_dir:
            save_dir = Path(self.trainer.log_dir) / "test_results"
        else:
            save_dir = Path("test_results")

        save_dir.mkdir(exist_ok=True, parents=True)
        print(f"\n💾 Saving test results to: {save_dir}")

        # Save arrays
        np.save(save_dir / "predictions_microstructure.npy", pred_micro.numpy())
        np.save(save_dir / "predictions_performance.npy", pred_perf.numpy())
        np.save(save_dir / "targets_microstructure.npy", target_micro.numpy())
        np.save(save_dir / "targets_performance.npy", target_perf.numpy())

        # Save report with metadata
        report["metadata"] = {
            "experiment_name": Path(self.logger.log_dir).parent.name,
            "version": self.logger.version,
            "log_dir": str(self.logger.log_dir),
        }

        with open(save_dir / "test_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Test results saved to: {save_dir}")

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

        # Highlights
        print("\n🎯 FEATURE PERFORMANCE HIGHLIGHTS:")

        all_r2s = {
            **{
                f"micro_{k}": v["R2"]
                for k, v in report["microstructure_metrics"]["per_feature"].items()
                if not np.isnan(v["R2"])
            },
            **{
                f"perf_{k}": v["R2"]
                for k, v in report["performance_metrics"]["per_feature"].items()
                if not np.isnan(v["R2"])
            },
        }

        if len(all_r2s) > 0:
            best_feature = max(all_r2s.items(), key=lambda x: x[1])
            worst_feature = min(all_r2s.items(), key=lambda x: x[1])

            print(f"  🏆 Best:  {best_feature[0]:<25} R² = {best_feature[1]:.4f}")
            print(f"  ⚠️  Worst: {worst_feature[0]:<25} R² = {worst_feature[1]:.4f}")

        print("=" * 100 + "\n")
