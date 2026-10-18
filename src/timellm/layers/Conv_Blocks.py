# ruff: noqa: N999, N801

"""Convolution blocks."""

import torch
from torch import nn


class Inception_Block_V1(nn.Module):
    """Inception Block V1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        *,
        init_weight: bool = True,
    ) -> None:
        """Initialize Inception block."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = [
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(self.num_kernels)
        ]
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize the weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        res_list = [self.kernels[i](x) for i in range(self.num_kernels)]
        return torch.stack(res_list, dim=-1).mean(-1)


class Inception_Block_V2(nn.Module):
    """Inception Block V2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        *,
        init_weight: bool = True,
    ) -> None:
        """Initialize Inception block."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[1, 2 * i + 3],
                    padding=[0, i + 1],
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=[2 * i + 3, 1],
                    padding=[i + 1, 0],
                )
            )
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize the weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        res_list = [self.kernels[i](x) for i in range(self.num_kernels + 1)]
        return torch.stack(res_list, dim=-1).mean(-1)
