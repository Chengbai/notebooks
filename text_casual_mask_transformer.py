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

        # self.norm = nn.LayerNorm(config.text_token_embedding)
        if mask is None:
            self.mask = torch.tril(torch.ones(config.max_text_len, config.max_text_len))
        else:
            self.mask = mask
        self.softmax = nn.Softmax(dim=-1)  # softmax accross the last dim
        self.out_proj = nn.Linear(
            config.text_token_embedding, config.text_token_embedding
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.wq.weight, std=self.wq.in_features**-0.5)
        nn.init.normal_(self.wk.weight, std=self.wk.in_features**-0.5)
        nn.init.normal_(self.wv.weight, std=self.wv.in_features**-0.5)

        std = self.out_proj.in_features**-0.5
        nn.init.normal_(self.out_proj.weight, std=std)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x TEXT_SEQ x TEXT_TOKEN_EMB
        """
        B, TEXT_SEQ, TEXT_TOKEN_EMB = x.size()
        # x_clone = x.clone()
        # x = self.norm(x)
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
        torch.masked_fill(attention, self.mask == 0, -torch.inf)
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
        # x = x_clone + vx
        output = self.out_proj(vx)
        return output


class TextTransformerBlock(nn.Module):
    def __init__(self, config: Config, mask: torch.tensor = None):
        super().__init__()
        self.config = config
        self.multihead_attention = TextMaskedMultiheadSelfAttention(
            config=config, mask=mask
        )
        self.norm1 = nn.LayerNorm(config.text_token_embedding)
        self.norm2 = nn.LayerNorm(config.text_token_embedding)

        # MLP
        self.linear1 = nn.Linear(
            config.text_token_embedding, 4 * config.text_token_embedding, bias=True
        )
        self.linear2 = nn.Linear(
            4 * config.text_token_embedding, config.text_token_embedding, bias=True
        )
        self.mlp = nn.Sequential(
            self.linear1,
            nn.GELU(),
            self.linear2,
            nn.Dropout(config.text_dropout),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.linear1.weight, std=self.linear1.in_features**-0.5)
        nn.init.normal_(self.linear2.weight, std=self.linear2.in_features**-0.5)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: batch text embedding, B x TEXT_SEQ x TEXT_EBM
        """
        residue = x

        x = self.norm1(x)  # B x TEXT_SEQ x TEXT_EBM
        x = self.multihead_attention(x)
        x += residue

        residue = x
        x = self.norm2(x)  # B x TEXT_SEQ x TEXT_EBM
        x = self.mlp(x)
        x = x + residue
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

    def forward(self, x: torch.tensor, need_embedding: bool = True) -> torch.tensor:
        """
        x: batch text embedding, B x TEXT_SEQ
        output: batch text embedding, B x TEXT_SEQ x TEXT_EBM
        """
        if need_embedding:
            x = self.text_token_embedding(x)
        return self.blocks(x)
