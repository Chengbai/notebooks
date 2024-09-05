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
        self.softmax = nn.Softmax(dim=-1)  # softmax accross the last dim
        self.out_proj = nn.Linear(
            config.img_patch_embedding, config.img_patch_embedding
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x IMG_PATCHES x IMG_PATCH_EMB
        """
        B, IMG_PATCHES, IMG_PATCH_EMB = x.size()
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
        output = self.out_proj(vx)
        return output


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
