import torch
import torch.nn as nn

from config import Config


class ImgMultiheadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.wq = nn.Linear(
            config.img_patch_embedding, config.img_patch_embedding
        )  # W = IMG_PATCH_EMB x IMG_PATCH_EMB
        self.wk = nn.Linear(
            config.img_patch_embedding, config.img_patch_embedding
        )  # W = IMG_PATCH_EMB x IMG_PATCH_EMB
        self.wv = nn.Linear(
            config.img_patch_embedding, config.img_patch_embedding
        )  # W = IMG_PATCH_EMB x IMG_PATCH_EMB
        self.norm = nn.LayerNorm(config.img_patch_embedding)
        self.softmax = nn.Softmax(dim=-1)  # softmax accross the last dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x IMG_PATCHES x IMG_PATCH_EMB
        """
        B, IMG_PATCHES, IMG_PATCH_EMB = x.size()
        x_clone = x.clone()
        x = self.norm(x)
        qx = self.wq(x)  # B x IMG_PATCHES x IMG_PATCH_EMB
        qx = qx.view(B, IMG_PATCHES, self.config.img_transformer_heads, -1).transpose(
            1, 2
        )  # B x IMG_HEADS x IMG_PATCHES x IMG_HEAD_EMB
        kx = self.wk(x)  # B x IMG_PATCHES x IMG_PATCH_EMB
        kx = kx.view(B, IMG_PATCHES, self.config.img_transformer_heads, -1).permute(
            0, 2, 3, 1
        )  # B x IMG_HEADS x IMG_HEAD_EMB x IMG_PATCHES
        attention = qx @ kx  # B x IMG_HEADS x IMG_PATCHES x IMG_PATCHES
        attention = attention / (
            IMG_PATCHES**0.5
        )  # B x IMG_HEADS x IMG_PATCHES x IMG_PATCHES
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
        x = x_clone + vx
        return x


class ImgTransformerBlock(nn.Module):
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
        self.multihead_attention = ImgMultiheadSelfAttention(config=config)
        self.norm = nn.LayerNorm(config.img_patch_embedding)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(
                config.img_patch_embedding, 4 * config.img_patch_embedding, bias=True
            ),
            nn.GELU(),
            nn.Linear(
                4 * config.img_patch_embedding, config.img_patch_embedding, bias=True
            ),
            nn.Dropout(config.img_dropout),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x IMG_PATCHES x IMG_PATCH_EMB
        """
        x = self.multihead_attention(x)

        x_clone = x.clone()
        x = self.norm(x)  # B x IMG_PATCHES x IMG_EMB
        x = self.mlp(x)
        x = x + x_clone
        return x


class ImgTransformer(nn.Module):
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
                ImgTransformerBlock(config=config)
                for _ in range(config.img_transformer_blocks)
            ]
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.blocks(x)
