# ruff: noqa: N801, N999

"""Autoformer Encoder/Decider layers."""

import torch
import torch.nn.functional as f
from torch import nn


class my_Layernorm(nn.Module):
    """Special designed layernorm for the seasonal part."""

    def __init__(self, channels: int | list) -> None:
        """Initialize my_Layernorm."""
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size: int, stride: int) -> None:
        """Initialize Moving Average block."""
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size: int) -> None:
        """Initialize Series decomposition block."""
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass."""
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """Multiple Series decomposition block from FEDformer."""

    def __init__(self, kernel_size: list) -> None:
        """Initialize Series decomp block."""
        super().__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass."""
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    """Autoformer encoder layer with the progressive decomposition architecture."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize EncoderLayer."""
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = f.relu if activation == "relu" else f.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple:
        """Forward pass."""
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """Autoformer encoder."""

    def __init__(
        self,
        attn_layers: list,
        conv_layers: list | None = None,
        norm_layer: torch.nn.Module | None = None,
    ) -> None:
        """Initialize Encoder."""
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple:
        """Forward pass."""
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(
                self.attn_layers, self.conv_layers, strict=False
            ):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """.Autoformer decoder layer with the progressive decomposition architecture."""

    def __init__(
        self,
        self_attention: torch.nn.Module,
        cross_attention: torch.nn.Module,
        d_model: int,
        c_out: int,
        d_ff: int | None = None,
        moving_avg: int = 25,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """Initialize DecoderLayer."""
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.activation = f.relu if activation == "relu" else f.gelu

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass."""
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        )
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(
            1, 2
        )
        return x, residual_trend


class Decoder(nn.Module):
    """Autoformer encoder."""

    def __init__(
        self,
        layers: list,
        norm_layer: torch.nn.Module | None = None,
        projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize the decoder."""
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
        trend: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass."""
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
