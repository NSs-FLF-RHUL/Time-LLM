# ruff: noqa: ARG002, N802, FBT001, FBT002, N999

"""Self attention family."""

from math import sqrt
from typing import Any

import numpy as np
import torch
from reformer_pytorch import LSHSelfAttention
from torch import nn

from timellm.utils.masking import ProbMask, TriangularCausalMask


class DSAttention(nn.Module):
    """De-stationary Attention."""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: float = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        *,
        output_attention: bool = False,
    ) -> None:
        """Initialize DSAttention layer."""
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass."""
        b, seq_len, h, e = queries.shape
        _, s, _, d = values.shape
        scale = self.scale or 1.0 / sqrt(e)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = (
            0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)
        )  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned
        # de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(b, seq_len, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        a = self.dropout(torch.softmax(scale * scores, dim=-1))
        v = torch.einsum("bhls,bshd->blhd", a, values)

        if self.output_attention:
            return v.contiguous(), a
        return v.contiguous(), None


class FullAttention(nn.Module):
    """Full Attention."""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: float = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        *,
        output_attention: bool = False,
    ) -> None:
        """Initialize Full Attention layer."""
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass."""
        b, seq_len, h, e = queries.shape
        _, s, _, d = values.shape
        scale = self.scale or 1.0 / sqrt(e)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(b, seq_len, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        a = self.dropout(torch.softmax(scale * scores, dim=-1))
        v = torch.einsum("bhls,bshd->blhd", a, values)

        if self.output_attention:
            return v.contiguous(), a
        return v.contiguous(), None


class ProbAttention(nn.Module):
    """Probability Attention."""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: float = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        *,
        output_attention: bool = False,
    ) -> None:
        """Initialize Probability Attention layer."""
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(
        self, q: torch.Tensor, k: torch.Tensor, sample_k: int, n_top: int
    ) -> tuple:  # n_top: c*ln(L_q)
        b, h, l_k, e = k.shape
        _, _, l_q, _ = q.shape

        # calculate the sampled Q_K
        k_expand = k.unsqueeze(-3).expand(b, h, l_q, l_k, e)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(l_k, (l_q, sample_k))
        k_sample = k_expand[:, :, torch.arange(l_q).unsqueeze(1), index_sample, :]
        q_k_sample = torch.matmul(q.unsqueeze(-2), k_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        m = q_k_sample.max(-1)[0] - torch.div(q_k_sample.sum(-1), l_k)
        m_top = m.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        q_reduce = q[
            torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], m_top, :
        ]  # factor*ln(L_q)
        q_k = torch.matmul(q_reduce, k.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return q_k, m_top

    def _get_initial_context(self, v: torch.Tensor, l_q: int) -> torch.Tensor:
        b, h, l_v, d = v.shape
        if not self.mask_flag:
            v_sum = v.mean(dim=-2)
            contex = v_sum.unsqueeze(-2).expand(b, h, l_q, v_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            if l_q != l_v:
                err_msg = f"l_q ({l_q}) and l_v ({l_v}) are different."
                raise ValueError(err_msg)
            contex = v.cumsum(dim=-2)
        return contex

    def _update_context(
        self,
        context_in: torch.Tensor,
        v: torch.Tensor,
        scores: torch.Tensor,
        index: torch.Tensor,
        l_q: int,
        attn_mask: torch.Tensor | None,
    ) -> tuple:
        b, h, l_v, d = v.shape

        if self.mask_flag:
            attn_mask = ProbMask(b, h, l_q, index, scores, device=v.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :
        ] = torch.matmul(attn, v).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([b, h, l_v, l_v]) / l_v).type_as(attn).to(attn.device)
            attns[
                torch.arange(b)[:, None, None], torch.arange(h)[None, :, None], index, :
            ] = attn
            return context_in, attns
        return context_in, None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass."""
        b, l_q, h, d = queries.shape
        _, l_k, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        u_part = self.factor * np.ceil(np.log(l_k)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(l_q)).astype("int").item()  # c*ln(L_q)

        u_part = min(l_k, u_part)
        u = min(l_q, u)

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=int(u_part), n_top=int(u)
        )

        # add scale factor
        scale = self.scale or 1.0 / sqrt(d)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, l_q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, l_q, attn_mask
        )

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ) -> None:
        """Initialize attention layer."""
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> tuple:
        """Forward pass."""
        b, seq_len, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads

        queries = self.query_projection(queries).view(b, seq_len, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(b, seq_len, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    """Reformer layer."""

    def __init__(
        self,
        attention: torch.nn.Module | None,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
        causal: bool = False,
        bucket_size: int = 4,
        n_hashes: int = 4,
    ) -> None:
        """Initialize ReformerLayer."""
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal,
        )

    def fit_length(self, queries: torch.Tensor) -> torch.Tensor:
        """Fit length."""
        # inside reformer: assert N % (bucket_size * 2) == 0
        b, n, c = queries.shape
        if n % (self.bucket_size * 2) == 0:
            return queries
        # fill the time series
        fill_len = (self.bucket_size * 2) - (n % (self.bucket_size * 2))
        return torch.cat(
            [queries, torch.zeros([b, fill_len, c]).to(queries.device)], dim=1
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        tau: torch.Tensor | None,
        delta: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass."""
        # in Reformer: defalut queries=keys
        queries = self.attn(self.fit_length(queries))[:, : queries.shape[1], :]
        return queries, None
