import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual block with optional SE attention"""

    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

        # Squeeze-and-Excitation (channel attention)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(out_channels, out_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels // 16, out_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionPooling3D(nn.Module):
    """Enhanced spatial attention pooling for 3D features"""

    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, 1),
            nn.BatchNorm3d(in_channels // 8),
            nn.ReLU(),
            nn.Conv3d(in_channels // 8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        attention_map = self.attention(x)  # [B, 1, D, H, W]
        weighted_features = x * attention_map  # [B, C, D, H, W]

        # Global average + max pooling
        avg_pool = weighted_features.mean(dim=[2, 3, 4])  # [B, C]
        max_pool = weighted_features.amax(dim=[2, 3, 4])  # [B, C]
        pooled = avg_pool + 0.5 * max_pool  # Combined pooling

        return pooled


class SimpleModel(nn.Module):
    """
    Enhanced model for large-scale battery prediction (780k samples).

    Key improvements for large datasets:
    - Deeper architecture with more capacity
    - SE attention blocks
    - Enhanced pooling strategies
    - Regularization tuned for large data

    Args:
        input_params_dim: Dimension of input parameters (default: 15)
        image_channels: Number of image channels (default: 1)
        micro_output_dim: Number of microstructure outputs (default: 4)
        perf_output_dim: Number of performance outputs (default: 5)
    """

    def __init__(
        self,
        input_params_dim: int = 15,
        image_channels: int = 1,
        micro_output_dim: int = 4,
        perf_output_dim: int = 5,
    ):
        super().__init__()

        # Store dimensions
        self.input_params_dim = input_params_dim
        self.micro_output_dim = micro_output_dim
        self.perf_output_dim = perf_output_dim

        # ========== 3D IMAGE ENCODER (DEEPER) ==========
        # Progressive encoding: 128³ → 64³ → 32³ → 16³ → 8³ → 4³
        self.image_encoder = nn.Sequential(
            # Stage 1: 128³ → 64³
            nn.Conv3d(
                image_channels, 32, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # Stage 2: 64³ → 32³ (2 blocks)
            ResidualBlock3D(32, 64, stride=2),
            ResidualBlock3D(64, 64),
            # Stage 3: 32³ → 16³ (3 blocks with SE)
            ResidualBlock3D(64, 128, stride=2),
            ResidualBlock3D(128, 128, use_se=True),
            ResidualBlock3D(128, 128),
            # Stage 4: 16³ → 8³ (3 blocks with SE)
            ResidualBlock3D(128, 256, stride=2),
            ResidualBlock3D(256, 256, use_se=True),
            ResidualBlock3D(256, 256),
            # Stage 5: 8³ → 4³ (4 blocks with SE)
            ResidualBlock3D(256, 512, stride=2),
            ResidualBlock3D(512, 512, use_se=True),
            ResidualBlock3D(512, 512),
            ResidualBlock3D(512, 512, use_se=True),
        )

        # Enhanced attention pooling
        self.image_attention_pool = AttentionPooling3D(512)

        # Additional image processing with residual connection
        self.image_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        # ========== PARAMETER ENCODER (DEEPER) ==========
        self.param_encoder = nn.Sequential(
            nn.Linear(input_params_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        # ========== CROSS-MODAL ATTENTION ==========
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, dropout=0.1, batch_first=True
        )

        # ========== FUSION MODULE (DEEPER) ==========
        self.fusion_dim = 512 + 512  # image + params

        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # ========== OUTPUT HEADS (DEEPER) ==========
        # Microstructure prediction head
        self.micro_head = nn.Sequential(
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, micro_output_dim),
        )

        # Performance prediction head
        self.perf_head = nn.Sequential(
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, perf_output_dim),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, params):
        """
        Forward pass with cross-modal attention

        Args:
            image: [B, 1, 128, 128, 128] - 3D microstructure images
            params: [B, 15] - Input parameters

        Returns:
            Dictionary with 'microstructure' [B, 4] and 'performance' [B, 5] predictions
        """
        # Encode image: [B, 1, 128, 128, 128] → [B, 512]
        img_features = self.image_encoder(image)  # [B, 512, 4, 4, 4]
        img_features = self.image_attention_pool(img_features)  # [B, 512]
        img_features = self.image_projection(img_features)  # [B, 512]

        # Encode parameters: [B, 15] → [B, 512]
        param_features = self.param_encoder(params)  # [B, 512]

        # Cross-modal attention (image attends to params)
        img_features_attn, _ = self.cross_attention(
            img_features.unsqueeze(1),  # [B, 1, 512]
            param_features.unsqueeze(1),  # [B, 1, 512]
            param_features.unsqueeze(1),  # [B, 1, 512]
        )
        img_features_attn = img_features_attn.squeeze(1)  # [B, 512]

        # Residual connection
        img_features = img_features + 0.5 * img_features_attn

        # Fuse modalities: [B, 1024] → [B, 512]
        fused_features = torch.cat([img_features, param_features], dim=1)  # [B, 1024]
        fused_features = self.fusion(fused_features)  # [B, 512]

        # Predict outputs
        micro_pred = self.micro_head(fused_features)  # [B, 4]
        perf_pred = self.perf_head(fused_features)  # [B, 5]

        return {
            "microstructure": micro_pred,
            "performance": perf_pred,
        }

    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
