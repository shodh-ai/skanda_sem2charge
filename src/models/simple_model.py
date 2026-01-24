import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple baseline model - just works!"""

    def __init__(self):
        super().__init__()

        # 3D CNN for images
        self.image_encoder = nn.Sequential(
            # Input: (B, 1, 128, 128, 128)
            nn.Conv3d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (B, 32, 64, 64, 64)
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (B, 64, 32, 32, 32)
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # -> (B, 128, 16, 16, 16)
            nn.Conv3d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # -> (B, 256, 1, 1, 1)
        )

        # MLP for input params
        self.param_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Fusion + outputs
        # 256 (image) + 128 (params) = 384
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Output heads
        self.micro_head = nn.Linear(256, 4)  # 4 microstructure outputs
        self.perf_head = nn.Linear(256, 5)  # 5 performance outputs

    def forward(self, image, params):
        # Encode image
        img_feat = self.image_encoder(image)
        img_feat = img_feat.flatten(1)  # (B, 256)

        # Encode params
        param_feat = self.param_encoder(params)  # (B, 128)

        # Fuse
        fused = torch.cat([img_feat, param_feat], dim=1)  # (B, 384)
        fused = self.fusion(fused)  # (B, 256)

        # Predict
        micro_pred = self.micro_head(fused)
        perf_pred = self.perf_head(fused)

        return {"microstructure": micro_pred, "performance": perf_pred}
