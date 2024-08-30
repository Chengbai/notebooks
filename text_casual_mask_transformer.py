from config import Config

from text_token_embedding import TextTokenEmbedding

import torch
import torch.nn as nn


class TextMaskedMultiheadSelfAttention(nn.Module):
    def __init__(self, config: Config, mask: torch.tensor = None):
        super().__init__()
        self.config = config
        self.wq = nn.Linear(
            config.text_token_embedding,
            config.text_token_embedding,
        )  # W = TEXT_SEQ x TEXT_EMB
        self.wk = nn.Linear(
            config.text_token_embedding, config.text_token_embedding
        )  # W = TEXT_SEQ x TEXT_EMB
        self.wv = nn.Linear(
            config.text_token_embedding, config.text_token_embedding
        )  # W = TEXT_SEQ x TEXT_EMB

        self.norm = nn.LayerNorm(config.text_token_embedding)
        if mask is None:
            self.mask = torch.tril(torch.ones(config.max_text_len, config.max_text_len))
        else:
            self.mask = mask
        self.softmax = nn.Softmax(dim=-1)  # softmax accross the last dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x TEXT_SEQ x TEXT_TOKEN_EMB
        """
        B, TEXT_SEQ, TEXT_TOKEN_EMB = x.size()
        x_clone = x.clone()
        x = self.norm(x)
        qx = self.wq(x)  # B x TEXT_SEQ x TEXT_TOKEN_EMB
        qx = qx.view(B, TEXT_SEQ, self.config.text_transformer_heads, -1).transpose(
            1, 2
        )  # B x TEXT_HEADS x TEXT_SEQ x TEXT_HEAD_EMB
        kx = self.wk(x)  # B x TEXT_SEQ x TEXT_TOKEN_EMB
        kx = kx.view(B, TEXT_SEQ, self.config.text_transformer_heads, -1).permute(
            0, 2, 3, 1
        )  # B x TEXT_HEADS x TEXT_HEAD_EMB x TEXT_SEQ
        attention = qx @ kx  # B x TEXT_HEADS x TEXT_SEQ x TEXT_SEQ
        attention = attention / (
            TEXT_SEQ**0.5
        )  # B x TEXT_HEADS x TEXT_HEAD_EMB x TEXT_HEAD_EMB

        self.mask = self.mask.to(attention.device)
        torch.masked_fill(attention, self.mask == 0, -float("inf"))
        attention = self.softmax(attention)

        vx = self.wv(x)  # B x TEXT_SEQ x TEXT_TOKEN_EMB
        vx = vx.view(B, TEXT_SEQ, self.config.text_transformer_heads, -1).transpose(
            1, 2
        )  # B x TEXT_HEADS x TEXT_SEQ x TEXT_EMB
        vx = attention @ vx  # B x TEXT_HEADS x TEXT_SEQ x TEXT_HEAD_EMB
        vx = vx.transpose(
            1, 2
        ).contiguous()  # B x TEXT_SEQ x TEXT_HEADS x TEXT_HEAD_EMB
        vx = vx.view(B, TEXT_SEQ, -1)  # B x TEXT_SEQ x TEXT_EMB
        x = x_clone + vx
        return x


class TextTransformerBlock(nn.Module):
    def __init__(self, config: Config, mask: torch.tensor = None):
        super().__init__()
        self.config = config
        self.multihead_attention = TextMaskedMultiheadSelfAttention(
            config=config, mask=mask
        )
        self.norm = nn.LayerNorm(config.text_token_embedding)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(
                config.text_token_embedding, 4 * config.text_token_embedding, bias=True
            ),
            nn.GELU(),
            nn.Linear(
                4 * config.text_token_embedding, config.text_token_embedding, bias=True
            ),
            nn.Dropout(config.text_dropout),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: batch text embedding, B x TEXT_SEQ x TEXT_EBM
        """
        x = self.multihead_attention(x)

        x_clone = x.clone()
        x = self.norm(x)  # B x TEXT_SEQ x TEXT_EBM
        x = self.mlp(x)
        x = x + x_clone
        return x


class TextMaskedTransformer(nn.Module):
    def __init__(self, config: Config, mask: torch.tensor = None):
        super().__init__()
        self.config = config
        self.text_token_embedding = TextTokenEmbedding(config=config)
        self.blocks = nn.Sequential(
            *[
                TextTransformerBlock(config=config, mask=mask)
                for _ in range(config.text_transformer_blocks)
            ]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: batch text embedding, B x TEXT_SEQ x TEXT_EBM
        """
        x = self.text_token_embedding(x)
        return self.blocks(x)
