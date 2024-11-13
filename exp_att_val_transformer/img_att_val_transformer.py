import math
import torch
import torch.nn as nn

from config import Config


class ImgAttValMultiheadSelfAttention(nn.Module):
    """
    - Single-head Attention
    x: B x IMG_PATCHES x IMG_PATCH_EMB
    wqk: IMG_PATCH_EMB x IMG_PATCH_EMB
    attention: [x] x [wqk] x [x.T]
        => [B x IMG_PATCHES x IMG_PATCH_EMB] x [IMG_PATCH_EMB x IMG_PATCH_EMB] x [B x IMG_PATCH_EMB x IMG_PATCHES]
        => [B x IMG_PATCHES x IMG_PATCH_EMB] x [B x IMG_PATCH_EMB x IMG_PATCHES]
        => [B x IMG_PATCHES x IMG_PATCHES]

    - Multi-heads Attention
    x: B x IMG_PATCHES x IMG_PATCH_EMB
    x_h: B x HEADS x IMG_PATCHES x HEAD_PATCH_EMB
    w_att_val: HEADS x HEAD_PATCH_EMB x HEAD_PATCH_EMB
    attention: [x_h] x [w_att_val] x [x_h.T]
        => [B x HEADS x IMG_PATCHES x HEAD_PATCH_EMB] x [HEADS x HEAD_PATCH_EMB x HEAD_PATCH_EMB] x [B x HEADS x HEAD_PATCH_EMB x IMG_PATCHES]
        => [B x HEADS x IMG_PATCHES x HEAD_PATCH_EMB] x [B x HEADS x HEAD_PATCH_EMB x IMG_PATCHES]
        => [B x HEADS x IMG_PATCHES x IMG_PATCHES]
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        assert self.config.img_patch_embedding % self.config.img_transformer_heads == 0
        HEAD_PATCH_EMB = (
            self.config.img_patch_embedding // self.config.img_transformer_heads
        )
        self.w_att_val = nn.Parameter(
            torch.empty(
                (self.config.img_transformer_heads, HEAD_PATCH_EMB, HEAD_PATCH_EMB)
            )
        )  # W = HEADS x HEAD_PATCH_EMB x HEAD_PATCH_EMB
        self.attention_bias = nn.Parameter(
            torch.empty(
                self.config.img_transformer_heads,
                self.config.img_patches,
                self.config.img_patches,
            )
        )
        self.wv = nn.Linear(
            config.img_patch_embedding, config.img_patch_embedding
        )  # W = IMG_PATCH_EMB x IMG_PATCH_EMB
        self.softmax = nn.Softmax(dim=-1)  # softmax accross the last dim
        self.out_proj = nn.Linear(
            config.img_patch_embedding, config.img_patch_embedding
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_uniform_(self.w_att_val, a=0.0)
        nn.init.normal_(self.attention_bias)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x IMG_PATCHES x IMG_PATCH_EMB
        """
        B, IMG_PATCHES, IMG_PATCH_EMB = x.size()
        HEAD_PATCH_EMB = IMG_PATCH_EMB // self.config.img_transformer_heads
        x_h = x.view(B, IMG_PATCHES, self.config.img_transformer_heads, -1).transpose(
            1, 2
        )  # B x IMG_HEADS x IMG_PATCHES x IMG_HEAD_EMB
        x_h_t = x_h.transpose(-1, -2)

        self.w_att_val = self.w_att_val.to(x.device)
        attention = x_h @ self.w_att_val  # B x IMG_HEADS x IMG_PATCHES x IMG_PATCHES
        attention = attention @ x_h_t  # B x IMG_HEADS x IMG_PATCHES x IMG_PATCHES
        attention = attention / (
            HEAD_PATCH_EMB**0.5
        )  # B x IMG_HEADS x IMG_PATCHES x IMG_PATCHES
        attention += self.attention_bias.to(x.device)
        attention = self.softmax(attention)

        vx = self.wv(x)  # B x IMG_PATCHES x IMG_PATCH_EMB
        vx = vx.view(B, IMG_PATCHES, self.config.img_transformer_heads, -1).transpose(
            1, 2
        )  # B x IMG_HEADS x IMG_PATCHES x IMG_HEAD_EMB
        vx = attention @ vx  # B x IMG_HEADS x IMG_PATCHES x IMG_HEAD_EMB
        vx = vx.transpose(
            1, 2
        ).contiguous()  # B x IMG_PATCHES x IMG_HEADS x IMG_HEAD_EMB
        vx = vx.view(B, IMG_PATCHES, -1)  # B x IMG_PATCHES x IMG_EMB
        output = self.out_proj(vx)
        return output


class ImgAttValTransformerBlock(nn.Module):
    """
    Transformer is a sequence to sequence model. Here we implement it as a decoder-only.
    Input:
        - x: tensor, B x TOKENS x TOKEN_EMB
    output:
        - y: tensor, B x TOKENS x TOKEN_EMB
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.multihead_attention = ImgAttValMultiheadSelfAttention(config=config)
        self.norm1 = nn.LayerNorm(config.img_patch_embedding)
        self.norm2 = nn.LayerNorm(config.img_patch_embedding)

        # MLP
        self.linear1 = nn.Linear(
            config.img_patch_embedding, 4 * config.img_patch_embedding, bias=True
        )

        self.linear2 = nn.Linear(
            4 * config.img_patch_embedding, config.img_patch_embedding, bias=True
        )

        self.mlp = nn.Sequential(
            self.linear1,
            nn.GELU(),
            self.linear2,
            nn.Dropout(config.img_dropout),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.linear1.weight, std=self.linear1.in_features**-0.5)
        nn.init.normal_(self.linear2.weight, std=self.linear2.in_features**-0.5)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x IMG_PATCHES x IMG_PATCH_EMB
        """
        residue = x
        x = self.norm1(x)  # B x IMG_PATCHES x IMG_EMB
        x = self.multihead_attention(x)
        x += residue

        residue = x
        x = self.norm2(x)  # B x IMG_PATCHES x IMG_EMB
        x = self.mlp(x)
        x += residue
        return x


class ImgAttValTransformer(nn.Module):
    """
    Transformer is a sequence to sequence model. Here we implement it as a decoder-only.
    Input:
        - x: tensor, B x TOKENS x TOKEN_EMB
    output:
        - y: tensor, B x TOKENS x TOKEN_EMB
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.blocks = nn.Sequential(
            *[
                ImgAttValTransformerBlock(config=config)
                for _ in range(config.img_transformer_blocks)
            ]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.blocks(x)
