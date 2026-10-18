# ruff: noqa: N999

"""Transformer Encoder/Decoder."""

from collections.abc import Callable

import torch
import torch.nn.functional as func
from torch import nn


class ConvLayer(nn.Module):
    """Convolution layer."""

    def __init__(self, c_in: int) -> None:
        """Initialize ConvLayer."""
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """Encoder Layer."""

    def __init__(
        self,
        attention: Callable,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize Encoder Layer."""
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = func.relu if activation == "relu" else func.gelu

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass."""
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """Encoder layer."""

    def __init__(
        self,
        attn_layers: list,
        conv_layers: list | None = None,
        norm_layer: nn.Module | None = None,
    ) -> None:
        """Initialize Encoder layer."""
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass."""
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers, strict=False)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """Decoder Layer."""

    def __init__(
        self,
        self_attention: nn.Module,
        cross_attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize Decoder Layer."""
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = func.relu if activation == "relu" else func.gelu

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """Transformer Decoder Layer."""

    def __init__(
        self,
        layers: list,
        norm_layer: torch.nn.Module | None = None,
        projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize Decoder."""
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
