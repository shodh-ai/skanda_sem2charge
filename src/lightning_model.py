import torch
import torch.nn as nn
import pytorch_lightning as pl
from .models.simple_model import SimpleModel


class BatteryLightningModel(pl.LightningModule):
    """Lightning wrapper for training"""

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

        # Log
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_micro_loss", micro_loss, on_epoch=True)
        self.log("train_perf_loss", perf_loss, on_epoch=True)

        return loss

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

        # Log
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_micro_loss", micro_loss)
        self.log("val_perf_loss", perf_loss)
        self.log("val_micro_mae", micro_mae)
        self.log("val_perf_mae", perf_mae)

        return loss

    def test_step(self, batch, batch_idx):
        pred = self(batch["image"], batch["input_params"])
        target = {
            "microstructure": batch["microstructure_outputs"],
            "performance": batch["performance_outputs"],
        }

        loss, micro_loss, perf_loss = self.compute_loss(pred, target)

        micro_mae = self.mae(pred["microstructure"], target["microstructure"])
        perf_mae = self.mae(pred["performance"], target["performance"])

        self.log("test_loss", loss)
        self.log("test_micro_loss", micro_loss)
        self.log("test_perf_loss", perf_loss)
        self.log("test_micro_mae", micro_mae)
        self.log("test_perf_mae", perf_mae)

        return loss

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
