import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from litdata import StreamingDataset
from pathlib import Path
import numpy as np


def collate_fn(batch):
    """Custom collate function to handle numpy arrays from StreamingDataset"""
    images = []
    input_params = []
    micro_outputs = []
    perf_outputs = []

    for item in batch:
        img = torch.from_numpy(np.array(item["image"])).unsqueeze(0).float()
        images.append(img)

        input_params.append(torch.from_numpy(np.array(item["input_params"])).float())
        micro_outputs.append(
            torch.from_numpy(np.array(item["microstructure_outputs"])).float()
        )
        perf_outputs.append(
            torch.from_numpy(np.array(item["performance_outputs"])).float()
        )

    return {
        "image": torch.stack(images),
        "input_params": torch.stack(input_params),
        "microstructure_outputs": torch.stack(micro_outputs),
        "performance_outputs": torch.stack(perf_outputs),
    }


class BatteryDataModule(pl.LightningDataModule):
    """DataModule for battery dataset using StreamingDataset"""

    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Setup streaming datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset = StreamingDataset(
                input_dir=str(self.data_dir / "train"), shuffle=True
            )
            self.val_dataset = StreamingDataset(
                input_dir=str(self.data_dir / "val"), shuffle=False
            )

        if stage == "test" or stage is None:
            self.test_dataset = StreamingDataset(
                input_dir=str(self.data_dir / "test"), shuffle=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
