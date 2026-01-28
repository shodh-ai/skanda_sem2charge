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


class BatteryLightningModel(pl.LightningModule):
    """Lightning wrapper for training with comprehensive metrics"""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Model
        self.model = SimpleModel()

        # Loss
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

        # Loss weights
        self.micro_weight = config["loss_weights"]["microstructure"]
        self.perf_weight = config["loss_weights"]["performance"]

        # === Training Metrics ===
        self.train_micro_r2 = R2Score(num_outputs=4)
        self.train_perf_r2 = R2Score(num_outputs=5)

        # === Validation Metrics ===
        self.val_micro_r2 = R2Score(num_outputs=4)
        self.val_perf_r2 = R2Score(num_outputs=5)

        # === Test Metrics ===
        self.test_micro_mse = MeanSquaredError()
        self.test_micro_mae = MeanAbsoluteError()
        self.test_micro_r2 = R2Score(num_outputs=4)
        self.test_micro_mape = MeanAbsolutePercentageError()

        self.test_perf_mse = MeanSquaredError()
        self.test_perf_mae = MeanAbsoluteError()
        self.test_perf_r2 = R2Score(num_outputs=5)
        self.test_perf_mape = MeanAbsolutePercentageError()

        # Store test predictions for detailed analysis
        self.test_predictions = []
        self.test_targets = []

    def forward(self, image, params):
        return self.model(image, params)

    def compute_loss(self, pred, target):
        micro_loss = self.mse(pred["microstructure"], target["microstructure"])
        perf_loss = self.mse(pred["performance"], target["performance"])
        total_loss = self.micro_weight * micro_loss + self.perf_weight * perf_loss
        return total_loss, micro_loss, perf_loss

    def training_step(self, batch, batch_idx):
        pred = self(batch["image"], batch["input_params"])
        target = {
            "microstructure": batch["microstructure_outputs"],
            "performance": batch["performance_outputs"],
        }

        loss, micro_loss, perf_loss = self.compute_loss(pred, target)

        # Update R2 metrics
        self.train_micro_r2(pred["microstructure"], target["microstructure"])
        self.train_perf_r2(pred["performance"], target["performance"])

        # Log
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("train_micro_loss", micro_loss, on_epoch=True, sync_dist=True)
        self.log("train_perf_loss", perf_loss, on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        # Log R2 scores
        self.log("train_micro_r2", self.train_micro_r2.compute(), sync_dist=True)
        self.log("train_perf_r2", self.train_perf_r2.compute(), sync_dist=True)

        # Reset metrics
        self.train_micro_r2.reset()
        self.train_perf_r2.reset()

    def validation_step(self, batch, batch_idx):
        pred = self(batch["image"], batch["input_params"])
        target = {
            "microstructure": batch["microstructure_outputs"],
            "performance": batch["performance_outputs"],
        }

        loss, micro_loss, perf_loss = self.compute_loss(pred, target)

        # Compute MAE
        micro_mae = self.mae(pred["microstructure"], target["microstructure"])
        perf_mae = self.mae(pred["performance"], target["performance"])

        # Update R2 metrics
        self.val_micro_r2(pred["microstructure"], target["microstructure"])
        self.val_perf_r2(pred["performance"], target["performance"])

        # Log
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_micro_loss", micro_loss, sync_dist=True)
        self.log("val_perf_loss", perf_loss, sync_dist=True)
        self.log("val_micro_mae", micro_mae, sync_dist=True)
        self.log("val_perf_mae", perf_mae, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        # Log R2 scores
        self.log("val_micro_r2", self.val_micro_r2.compute(), sync_dist=True)
        self.log("val_perf_r2", self.val_perf_r2.compute(), sync_dist=True)

        # Reset metrics
        self.val_micro_r2.reset()
        self.val_perf_r2.reset()

    def test_step(self, batch, batch_idx):
        pred = self(batch["image"], batch["input_params"])
        target = {
            "microstructure": batch["microstructure_outputs"],
            "performance": batch["performance_outputs"],
        }

        loss, micro_loss, perf_loss = self.compute_loss(pred, target)

        micro_mae = self.mae(pred["microstructure"], target["microstructure"])
        perf_mae = self.mae(pred["performance"], target["performance"])

        # Update all test metrics
        self.test_micro_mse(pred["microstructure"], target["microstructure"])
        self.test_micro_mae(pred["microstructure"], target["microstructure"])
        self.test_micro_r2(pred["microstructure"], target["microstructure"])
        self.test_micro_mape(pred["microstructure"], target["microstructure"])

        self.test_perf_mse(pred["performance"], target["performance"])
        self.test_perf_mae(pred["performance"], target["performance"])
        self.test_perf_r2(pred["performance"], target["performance"])
        self.test_perf_mape(pred["performance"], target["performance"])

        # Store predictions for detailed analysis
        self.test_predictions.append(
            {
                "microstructure": pred["microstructure"].detach().cpu(),
                "performance": pred["performance"].detach().cpu(),
            }
        )
        self.test_targets.append(
            {
                "microstructure": target["microstructure"].detach().cpu(),
                "performance": target["performance"].detach().cpu(),
            }
        )

        # Log basic metrics
        self.log("test_loss", loss)
        self.log("test_micro_loss", micro_loss)
        self.log("test_perf_loss", perf_loss)
        self.log("test_micro_mae", micro_mae)
        self.log("test_perf_mae", perf_mae)

        return loss

    def on_test_epoch_end(self):
        """Compute comprehensive test metrics and create report"""

        # Compute all metrics
        test_micro_mse = self.test_micro_mse.compute()
        test_micro_mae = self.test_micro_mae.compute()
        test_micro_r2 = self.test_micro_r2.compute()
        test_micro_mape = self.test_micro_mape.compute()
        test_micro_rmse = torch.sqrt(test_micro_mse)

        test_perf_mse = self.test_perf_mse.compute()
        test_perf_mae = self.test_perf_mae.compute()
        test_perf_r2 = self.test_perf_r2.compute()
        test_perf_mape = self.test_perf_mape.compute()
        test_perf_rmse = torch.sqrt(test_perf_mse)

        # Log comprehensive metrics
        self.log("test_micro_mse", test_micro_mse)
        self.log("test_micro_rmse", test_micro_rmse)
        self.log("test_micro_r2", test_micro_r2)
        self.log("test_micro_mape", test_micro_mape)

        self.log("test_perf_mse", test_perf_mse)
        self.log("test_perf_rmse", test_perf_rmse)
        self.log("test_perf_r2", test_perf_r2)
        self.log("test_perf_mape", test_perf_mape)

        # Create comprehensive report
        report = {
            "model_info": {
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            },
            "microstructure_metrics": {
                "MSE": float(test_micro_mse),
                "RMSE": float(test_micro_rmse),
                "MAE": float(test_micro_mae),
                "R2_Score": float(test_micro_r2),
                "MAPE": float(test_micro_mape),
            },
            "performance_metrics": {
                "MSE": float(test_perf_mse),
                "RMSE": float(test_perf_rmse),
                "MAE": float(test_perf_mae),
                "R2_Score": float(test_perf_r2),
                "MAPE": float(test_perf_mape),
            },
            "overall_metrics": {
                "total_loss": float(test_micro_mse + test_perf_mse),
                "weighted_loss": float(
                    self.micro_weight * test_micro_mse
                    + self.perf_weight * test_perf_mse
                ),
            },
        }

        # Store report for later saving
        self.test_report = report

        # Print detailed console report
        self._print_test_report(report)

    def _print_test_report(self, report):
        """Print formatted test report to console"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS")
        print("=" * 80)

        print("\n🏗️  MODEL INFORMATION:")
        print(
            f"  ├─ Total Parameters:      {report['model_info']['total_parameters']:,}"
        )
        print(
            f"  └─ Trainable Parameters:  {report['model_info']['trainable_parameters']:,}"
        )

        print("\n🔬 MICROSTRUCTURE PREDICTIONS:")
        m = report["microstructure_metrics"]
        print(f"  ├─ MSE:   {m['MSE']:.6f}")
        print(f"  ├─ RMSE:  {m['RMSE']:.6f}")
        print(f"  ├─ MAE:   {m['MAE']:.6f}")
        print(f"  ├─ R²:    {m['R2_Score']:.6f}")
        print(f"  └─ MAPE:  {m['MAPE']:.4f}%")

        print("\n⚡ PERFORMANCE PREDICTIONS:")
        p = report["performance_metrics"]
        print(f"  ├─ MSE:   {p['MSE']:.6f}")
        print(f"  ├─ RMSE:  {p['RMSE']:.6f}")
        print(f"  ├─ MAE:   {p['MAE']:.6f}")
        print(f"  ├─ R²:    {p['R2_Score']:.6f}")
        print(f"  └─ MAPE:  {p['MAPE']:.4f}%")

        print("\n📈 OVERALL METRICS:")
        o = report["overall_metrics"]
        print(f"  ├─ Total Loss:    {o['total_loss']:.6f}")
        print(f"  └─ Weighted Loss: {o['weighted_loss']:.6f}")

        print("=" * 80 + "\n")

    def configure_optimizers(self):
        opt_cfg = self.config["optimizer"]

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config["scheduler"]["factor"],
            patience=self.config["scheduler"]["patience"],
            min_lr=self.config["scheduler"]["min_lr"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
