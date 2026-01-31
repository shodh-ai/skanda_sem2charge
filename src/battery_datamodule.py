import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from litdata import StreamingDataset
from pathlib import Path
import numpy as np
import json
from typing import Optional

from .utils import (
    INPUT_FEATURES,
    MICROSTRUCTURE_FEATURES,
    PERFORMANCE_FEATURES,
    get_loss_weight_tensors,
)


def collate_fn(batch):
    """Custom collate function to handle numpy arrays from StreamingDataset"""
    images = []
    input_params = []
    micro_outputs = []
    perf_outputs = []
    sample_ids = []
    param_ids = []

    for item in batch:
        # Image: [1, 128, 128, 128]
        img = torch.from_numpy(np.array(item["image"])).unsqueeze(0).float()
        images.append(img)

        # Input parameters: [15]
        input_params.append(torch.from_numpy(np.array(item["input_params"])).float())

        # Microstructure outputs: [4]
        micro_outputs.append(
            torch.from_numpy(np.array(item["microstructure_outputs"])).float()
        )

        # Performance outputs: [5]
        perf_outputs.append(
            torch.from_numpy(np.array(item["performance_outputs"])).float()
        )

        # Metadata for tracking
        sample_ids.append(item["sample_id"])
        param_ids.append(item["param_id"])

    return {
        "image": torch.stack(images),  # [B, 1, 128, 128, 128]
        "input_params": torch.stack(input_params),  # [B, 15]
        "microstructure_outputs": torch.stack(micro_outputs),  # [B, 4]
        "performance_outputs": torch.stack(perf_outputs),  # [B, 5]
        "sample_ids": sample_ids,
        "param_ids": param_ids,
    }


class BatteryDataModule(pl.LightningDataModule):
    """
    DataModule for battery dataset using StreamingDataset with multi-task support.

    Features:
    - Automatic loading of normalization statistics
    - Configurable loss weights for multi-task learning
    - Feature names mapping for interpretability
    - Support for weighted sampling (optional)
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.performance_weights, self.microstructure_weights = (
            get_loss_weight_tensors()
        )

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Normalization statistics
        self.norm_stats = None
        self._load_normalization_stats()

    def _load_normalization_stats(self):
        """Load normalization statistics for denormalization during inference"""
        stats_file = self.data_dir / "normalization_stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.norm_stats = json.load(f)
            print(f"✅ Loaded normalization stats from {stats_file}")
        else:
            print(f"⚠️ Warning: normalization_stats.json not found at {stats_file}")

    def setup(self, stage: Optional[str] = None):
        """Setup streaming datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = StreamingDataset(
                input_dir=str(self.data_dir / "train"),
                shuffle=True,
                drop_last=True,
            )
            self.val_dataset = StreamingDataset(
                input_dir=str(self.data_dir / "val"), shuffle=False
            )
            print(f"📊 Train samples: {len(self.train_dataset):,}")
            print(f"📊 Val samples:   {len(self.val_dataset):,}")

        if stage == "test" or stage is None:
            self.test_dataset = StreamingDataset(
                input_dir=str(self.data_dir / "test"), shuffle=False
            )
            print(f"📊 Test samples:  {len(self.test_dataset):,}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )

    def get_feature_names(self):
        """Returns feature names from utils"""
        return {
            "input": INPUT_FEATURES,
            "microstructure": MICROSTRUCTURE_FEATURES,
            "performance": PERFORMANCE_FEATURES,
        }
