"""Masks for Transformer and Informer architectures."""

import numpy as np
import torch


class TriangularCausalMask:
    """
    Causal attention mask.

     Attention masks are used in Transformer models to prevent tokens from attending
     to future positions.
    """

    def __init__(self, b: int, l: int, device: str | torch.device = "cpu") -> None:  # noqa: E741
        """
        Initialize the mask.

        :param b: batch size
        :param l: sequence length
        :param device: device (cpu or cuda)
        """
        mask_shape = [b, 1, l, l]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """Return the mask."""
        return self._mask


class ProbMask:
    """
    Sparse causal attention mask.

    Sparse casual attention masks are used in the Informer architecture.
    """

    def __init__(
        self,
        b: int,
        h: int,
        l: int,  # noqa: E741
        index: np.ndarray | torch.Tensor,
        scores: np.ndarray | torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Initialize the mask.

        :param b: batch size
        :param h: number of attention heads
        :param l: sequence length
        :param index: indices to compute the mask at
        :param scores: ???
        :param device: device (cpu or cuda)
        """
        _mask = torch.ones(l, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(b, h, l, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> torch.Tensor:
        """Return the mask."""
        return self._mask
