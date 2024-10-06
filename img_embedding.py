import torch
import torch.nn as nn

from config import Config


class ImageEmbedding(nn.Module):
    """
    Encode a batch of images into a tensor: BATCH x IMG_PATCH_SEQ x IMG_PATCH_EMBEDDING
    """

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None
        assert (
            config.img_h_size % config.img_patch_size == 0
        ), f"img height {config.img_h_size} is not evenly splitable by patch height: {config.img_patch_size}"
        assert (
            config.img_w_size % config.img_patch_size == 0
        ), f"img width {config.img_w_size} is not evenly splitable by patch width: {config.img_patch_size}"
        self.config = config

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=config.img_patch_embedding,
            kernel_size=config.img_patch_size,
            stride=config.img_patch_size,
            padding="valid",
            bias=True,
        )

        self.pos_embedding = nn.Embedding(
            config.img_patches, config.img_patch_embedding
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.conv.weight, std=0.2)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x: B x C x H x W tensor
        output: B x IMG_PATCH_SEQ x IMG_PATCH_EMB
        """
        x = self.conv(x)  # => B, IMG_PATCH_EMB x H_IMG_PATCHES x W_IMG_PATCHES
        B, IMG_PATCH_EMB, _, _ = x.size()
        x = x.view(B, IMG_PATCH_EMB, -1)  # => B x IMG_PATCH_EMB x IMG_PATCH_SEQ
        x = x.permute(0, 2, 1)  # => B x IMG_PATCH_SEQ x IMG_PATCH_EMB
        x += self.pos_embedding(
            torch.arange(self.config.img_patches).to(device=x.device)
        )

        return x
