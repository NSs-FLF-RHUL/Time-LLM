# ruff: noqa: N999, D100, D102

from argparse import Namespace

import torch
import torch.nn.functional as func
from torch import nn

from timellm.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from timellm.layers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)
from timellm.layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection.

    With inherent O(LlogL) complexity. Paper
    link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs: Namespace) -> None:
        """Initialize Autoformer model."""
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for dl in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )
        # Decoder
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.dec_embedding = DataEmbedding_wo_pos(
                configs.dec_in,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                mask_flag=True,
                                factor=configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                mask_flag=False,
                                factor=configs.factor,
                                attention_dropout=configs.dropout,
                                output_attention=False,
                            ),
                            configs.d_model,
                            configs.n_heads,
                        ),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for dl in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True),
            )
        if self.task_name == "imputation":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = func.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        # final
        return trend_part + seasonal_part

    def imputation(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,  # noqa: ARG002
        x_mark_dec: torch.Tensor,  # noqa: ARG002
        mask: torch.Tensor | None,  # noqa: ARG002
    ) -> torch.Tensor:
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        return self.projection(enc_out)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        return self.projection(enc_out)

    def classification(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor
    ) -> torch.Tensor:
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)  # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)  # (batch_size, num_classes)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Forward pass."""
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == "imputation":
            return self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask
            )  # [B, L, D]
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)  # [B, L, D]
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)  # [B, N]
        return None
