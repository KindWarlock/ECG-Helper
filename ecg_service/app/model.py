from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
MAMBA_DIR = ROOT_DIR / "mamba"
if str(MAMBA_DIR) not in sys.path:
    sys.path.insert(0, str(MAMBA_DIR))

from mamba_ssm import Mamba2  # noqa: E402


class MambaECGBlock(torch.nn.Module):
    """Residual Mamba block for sequence features shaped as (batch, length, hidden)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class ECGConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel: int = 3,
        conv_stride: int = 1,
        pool_kernel: int = 4,
        pool_stride: int = 2,
        groups: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel,
            stride=conv_stride,
            groups=groups,
            padding=padding,
        )
        self.norm = torch.nn.GroupNorm(groups, out_channels)
        self.activation = torch.nn.GELU()
        self.pool = torch.nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class MambaECGClassifier(torch.nn.Module):
    """Mamba-based multilabel classifier compatible with the saved notebook checkpoint."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 20,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.5,
        conv_blocks_config: list[dict] | None = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model

        if conv_blocks_config is None:
            conv_blocks_config = [
                dict(conv_kernel=101, conv_stride=10, pool_kernel=4, pool_stride=2),
                dict(conv_kernel=15, conv_stride=1, pool_kernel=4, pool_stride=1),
            ]

        conv_blocks = []
        in_channels = input_channels
        out_channels = input_channels * d_model
        for block_config in conv_blocks_config:
            conv_blocks.append(
                ECGConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_kernel=block_config.get("conv_kernel", 3),
                    conv_stride=block_config.get("conv_stride", 1),
                    pool_kernel=block_config.get("pool_kernel", 4),
                    pool_stride=block_config.get("pool_stride", 2),
                    groups=input_channels,
                    padding=block_config.get("padding", 0),
                )
            )
            in_channels = out_channels

        self.conv_blocks = torch.nn.Sequential(*conv_blocks)
        self.blocks = torch.nn.ModuleList(
            [
                MambaECGBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.lead_embedding = torch.nn.Parameter(torch.zeros(1, input_channels, 1, d_model))
        torch.nn.init.normal_(self.lead_embedding, std=0.02)

        self.channel_attention_query = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.channel_attention = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=1,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(d_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = x.shape
        x = self.conv_blocks(x)
        samples = x.shape[-1]

        x = x.reshape(batch_size, channels, self.d_model, samples)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch_size * channels, samples, self.d_model)

        for block in self.blocks:
            x = block(x)

        x = x.reshape(batch_size, channels, samples, self.d_model)
        x = x + self.lead_embedding[:, :channels, :, :]
        x = x.mean(dim=2)

        query = self.channel_attention_query.expand(batch_size, -1, -1)
        x, _ = self.channel_attention(query, x, x)
        x = self.norm(x.squeeze(1))
        return self.classifier(x)
