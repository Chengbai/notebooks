import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_util import get_logger
from config import Config
from datetime import datetime
from img_embedding import ImageEmbedding
from img_transformer import ImgTransformer
from img_util import show_img_tensor_CHW
from loss import constrastive_logit_loss
from fliker_comment_tokenizer import FlikerCommentTokenizer
from img_comment_dataset import ImgCommentDataset
from model_util import count_parameters
from pathlib import Path
from text_token_embedding import TextTokenEmbedding
from exp_att_val_transformer.text_casual_mask_att_val_transformer import (
    TextAttValMaskedTransformer,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF


logger = get_logger(__name__)


class ImgCaptionAttValTransformerModel(nn.Module):
    def __init__(self, config: Config, tokenizer):
        super().__init__()

        assert config is not None
        assert (
            config.img_patch_embedding == config.text_token_embedding
        ), f"img_patch_embedding: {config.img_patch_embedding} should be same as text_token_embedding: {config.text_token_embedding}"

        self.config = config
        self.tokenizer = tokenizer

        # ---------------------------------------------------------
        # [<imge> x IMG_PATCHES][<bos>][TEXT_TOKEN_EMB x N][<pad>*]
        # ---------------------------------------------------------
        #           i0  i1  b   t1  t2  t3
        # <image> | 1,  1,  1,  1,  1,  1 |
        # <image> | 1,  1,  1,  1,  1,  1 |
        # <bos>   | 1,  1,  1,  1,  1,  1 |
        # t1      | 1,  1,  1,  1,  0,  0 |
        # t2      | 1,  1,  1,  1,  1,  0 |
        # t3      | 1,  1,  1,  1,  1,  1 |
        # ---------------------------------------------------------
        # NOTE: <bos> token seems introduced confusing. Removed for now.
        # ---------------------------------------------------------

        self.img_bos_mask = torch.ones(
            size=(config.img_patches, config.img_patches + config.max_text_len)
        )  # (IMG_PATCHES) x (IMG_PATCHES + TEXT_TOKENS)
        self.text_mask = torch.hstack(
            [
                torch.ones(size=(config.max_text_len, config.img_patches)),
                torch.tril(
                    torch.ones(size=(config.max_text_len, config.max_text_len)),
                    diagonal=0,
                ),
            ]
        )
        self.mask = torch.vstack([self.img_bos_mask, self.text_mask])
        self.transformer = TextAttValMaskedTransformer(config=config, mask=self.mask)
        self.lm_head = nn.Linear(
            config.text_token_embedding, tokenizer.vocab_size, bias=False
        )
        # B x tokens x token_emb @ token_emb x vocab_size => B x tokens x vocab_size

    def forward(
        self,
        img_feature: torch.tensor,
        text_feature: torch.tensor,
        text_mask: torch.tensor,
        batch_target_text_token: torch.tensor,
        target_text_mask_tensor: torch.tensor,
        bos_embedding: torch.tensor,
    ):
        """
        inputs:
            - img_feature: B x IMG_PATCHES x IMG_PATCH_EMB
            - text_feature: B x TEXT_TOKEN x TEXT_EMB
            - text_mask: B x TEXT_TOKEN x 1
            - batch_target_text_token: B x TEXT_TOKEN
        outputs:
            - text prediction:
            - loss
        """
        bos_embedding = bos_embedding.view(1, 1, -1)  # 1 x 1 x TEXT_EMB
        bos_embedding = bos_embedding.to(img_feature.device)
        assert (
            len(img_feature.size())
            == len(bos_embedding.size())
            == len(text_feature.size())
        )
        bos_embedding_ext = bos_embedding.expand(img_feature.size()[0], -1, -1)
        x = torch.cat(
            [img_feature, text_feature],
            dim=1,
        )  # B x [IMG_PATCHES + 1 + TEXT_TOKEN] x IMG_PATCH_EMB
        x = self.transformer(
            x=x, need_embedding=False
        )  # B x [IMG_PATCHES + 1 + TEXT_TOKEN] x IMG_PATCH_EMB
        x = self.lm_head(x)  # B x [IMG_PATCHES + 1 + TEXT_TOKEN] x vocab_size

        # extract the last `[self.config.max_text_len - 1:-1]` token positions.
        # `-1` here to align the position to `[IMG_PATCHES + 1]`. Predict current from 1-step before info
        text_pos_mask = torch.arange(
            start=-self.config.max_text_len - 1, end=-1, step=1
        )
        batch_text_logits = x[:, text_pos_mask, :]  # B x TEXT_TOKEN x vocab_size

        B, TEXT_TOKEN, vocab_size = batch_text_logits.size()

        ############################################################################
        valid_target_text_token_length, valid_target_text_index = torch.max(
            target_text_mask_tensor, dim=1, keepdim=False
        )
        # logger.info(f"target_text_tokens: {target_text_tokens}")
        batch_text_loss = torch.tensor(0.0, device=batch_target_text_token.device)
        for bi, token_length in zip(
            torch.arange(B, device=batch_target_text_token.device),
            valid_target_text_token_length,
        ):
            if token_length == 0:
                # this is called at the beginning of create_caption_from_aug_img_tensor
                continue
            target_text_logits = batch_text_logits[bi][:token_length]
            target_text_token = batch_target_text_token[bi][:token_length]
            if self.training and target_text_token[0] != 2:
                # At training time, expect all of the text token from dataloader will start with `<bos>` which has value 2.
                raise Exception(
                    f"batch_target_text_token[bi]: {batch_target_text_token[bi]}, target_text_token: {target_text_token}"
                )
            target_text_loss = F.cross_entropy(
                target_text_logits, target_text_token, reduction="mean"
            )
            batch_text_loss += target_text_loss

            if self.training and self.config.debugging:
                target_caption = self.tokenizer.decode(target_text_token)
                prdict_token_max = torch.argmax(target_text_logits, dim=-1)
                prdict_caption = self.tokenizer.decode(prdict_token_max)
                logger.info(f"target_text_token: {target_text_token.size()}")
                logger.info(f"target_text_logits: {target_text_logits.size()}")
                logger.info(f"target_caption: {target_caption}")
                logger.info(f"prdict_caption: {prdict_caption}")

        batch_text_loss = batch_text_loss / torch.tensor(
            B, device=batch_target_text_token.device
        )
        ############################################################################

        # batch_text_logits = batch_text_logits.view(B * TEXT_TOKEN, -1)
        # batch_target_text_token = batch_target_text_token.view(B * TEXT_TOKEN)
        # batch_text_loss = F.cross_entropy(
        #     batch_text_logits, batch_target_text_token, reduction="mean"
        # )

        return batch_text_logits, batch_text_loss
