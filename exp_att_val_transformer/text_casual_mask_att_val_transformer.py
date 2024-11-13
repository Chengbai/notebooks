import math
import torch
import torch.nn as nn

from config import Config

from text_token_embedding import TextTokenEmbedding


class TextAttValMaskedMultiheadSelfAttention(nn.Module):
    """
    - Single-head Attention
    x: B x TEXT_SEQ x TEXT_EMB
    wqk: TEXT_EMB x TEXT_EMB
    attention: [x] x [wqk] x [x.T]
        => [B x TEXT_SEQ x TEXT_EMB] x [TEXT_EMB x TEXT_EMB] x [B x TEXT_EMB x TEXT_SEQ]
        => [B x TEXT_SEQ x TEXT_EMB] x [B x TEXT_EMB x TEXT_SEQ]
        => [B x TEXT_SEQ x TEXT_SEQ]

    - Multi-heads Attention
    x: B x TEXT_SEQ x TEXT_EMB
    x_h: B x HEADS x TEXT_SEQ x HEAD_PATCH_EMB
    w_att_val: HEADS x HEAD_PATCH_EMB x HEAD_PATCH_EMB
    attention: [x_h] x [w_att_val] x [x_h.T]
        => [B x HEADS x TEXT_SEQ x HEAD_PATCH_EMB] x [HEADS x HEAD_PATCH_EMB x HEAD_PATCH_EMB] x [B x HEADS x HEAD_PATCH_EMB x TEXT_SEQ]
        => [B x HEADS x TEXT_SEQ x HEAD_PATCH_EMB] x [B x HEADS x HEAD_PATCH_EMB x TEXT_SEQ]
        => [B x HEADS x TEXT_SEQ x TEXT_SEQ]
    """

    def __init__(self, config: Config, mask: torch.tensor = None):
        super().__init__()
        self.config = config

        assert (
            self.config.text_token_embedding % self.config.text_transformer_heads == 0
        )
        HEAD_PATCH_EMB = (
            self.config.text_token_embedding // self.config.text_transformer_heads
        )
        self.w_att_val = nn.Parameter(
            torch.empty(
                (self.config.text_transformer_heads, HEAD_PATCH_EMB, HEAD_PATCH_EMB)
            )
        )  # W = HEADS x HEAD_PATCH_EMB x HEAD_PATCH_EMB

        self.wv = nn.Linear(
            config.text_token_embedding, config.text_token_embedding
        )  # W = TEXT_SEQ x TEXT_EMB

        # self.norm = nn.LayerNorm(config.text_token_embedding)
        if mask is None:
            """
            tensor([[1., 0., 0., 0.],
                    [1., 1., 0., 0.],
                    [1., 1., 1., 0.],
                    [1., 1., 1., 1.]])
            => mask-fill
            tensor([[1., -inf, -inf, -inf],
                    [1., 1., -inf, -inf],
                    [1., 1., 1., -inf],
                    [1., 1., 1., 1.]])
            => softmax
            tensor([[1.0000, 0.0000, 0.0000, 0.0000],
                    [0.5000, 0.5000, 0.0000, 0.0000],
                    [0.3333, 0.3333, 0.3333, 0.0000],
                    [0.2500, 0.2500, 0.2500, 0.2500]])
            """
            self.mask = torch.tril(
                torch.ones(config.max_text_len, config.max_text_len), diagonal=0
            )
        else:
            self.mask = mask
        self.softmax = nn.Softmax(dim=-1)  # softmax accross the last dim
        self.out_proj = nn.Linear(
            config.text_token_embedding, config.text_token_embedding
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.w_att_val, a=0.0)
        nn.init.normal_(self.wv.weight, std=self.wv.in_features**-0.5)

        std = self.out_proj.in_features**-0.5
        nn.init.normal_(self.out_proj.weight, std=std)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x TEXT_SEQ x TEXT_TOKEN_EMB
        """
        B, TEXT_SEQ, TEXT_TOKEN_EMB = x.size()
        HEAD_PATCH_EMB = (
            self.config.text_token_embedding // self.config.text_transformer_heads
        )
        x_h = x.view(B, TEXT_SEQ, self.config.text_transformer_heads, -1).transpose(
            1, 2
        )  # B x HEADS x TEXT_SEQ x HEAD_PATCH_EMB
        x_h_t = x_h.transpose(-1, -2)  # B x HEADS x HEAD_PATCH_EMB x TEXT_SEQ

        self.w_att_val = self.w_att_val.to(x.device)
        attention = x_h @ self.w_att_val  # B x HEADS x TEXT_SEQ x HEAD_PATCH_EMB
        attention = attention @ x_h_t  # B x HEADS x TEXT_SEQ x TEXT_SEQ
        attention = attention / (HEAD_PATCH_EMB**0.5)  # B x HEADS x TEXT_SEQ x TEXT_SEQ

        self.mask = self.mask.to(x.device)
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


class TextAttValTransformerBlock(nn.Module):
    def __init__(self, config: Config, mask: torch.tensor = None):
        super().__init__()
        self.config = config
        self.multihead_attention = TextAttValMaskedMultiheadSelfAttention(
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


class TextAttValMaskedTransformer(nn.Module):
    def __init__(self, config: Config, mask: torch.tensor = None):
        super().__init__()
        self.config = config
        self.text_token_embedding = TextTokenEmbedding(config=config)

        self.blocks = nn.Sequential(
            *[
                TextAttValTransformerBlock(config=config, mask=mask)
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
