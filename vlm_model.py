import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from datetime import datetime
from img_embedding import ImageEmbedding
from img_transformer import ImgTransformer
from img_util import show_img_tensor_CHW
from loss import constrastive_logit_loss
from fliker_comment_tokenizer import FlikerCommentTokenizer
from fliker_img_comment_dataset import ImgCommentDataset
from model_util import count_parameters
from pathlib import Path
from text_token_embedding import TextTokenEmbedding
from text_casual_mask_transformer import TextMaskedTransformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF


class ImgCaptionModel(nn.Module):
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
        # t1      | 1,  1,  1,  0,  0,  0 |
        # t2      | 1,  1,  1,  1,  0,  0 |
        # t3      | 1,  1,  1,  1,  1,  0 |
        self.img_bos_mask = torch.ones(
            size=(config.img_patches + 1, config.img_patches + 1 + config.max_text_len)
        )  # (IMG_PATCHES + 1) x (IMG_PATCHES + 1 + TEXT_TOKENS)
        self.text_mask = torch.hstack(
            [
                torch.ones(size=(config.max_text_len, config.img_patches + 1)),
                torch.tril(torch.ones(size=(config.max_text_len, config.max_text_len)))
                - torch.eye(config.max_text_len),
            ]
        )
        self.mask = torch.vstack([self.img_bos_mask, self.text_mask])
        self.transformer = TextMaskedTransformer(config=config, mask=self.mask)
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
            [img_feature, bos_embedding_ext, text_feature], dim=1
        )  # B x [IMG_PATCHES + 1 + TEXT_TOKEN] x IMG_PATCH_EMB
        x = self.transformer(
            x=x, need_embedding=False
        )  # B x [IMG_PATCHES + 1 + TEXT_TOKEN] x IMG_PATCH_EMB
        x = self.lm_head(x)  # B x [IMG_PATCHES + 1 + TEXT_TOKEN] x vocab_size

        if batch_target_text_token is None:
            loss = None
        else:
            # extract the last `self.config.max_text_len` token positions
            text_pos_mask = torch.arange(start=-self.config.max_text_len, end=0, step=1)
            batch_text_logits = x[:, text_pos_mask, :]  # B x TEXT_TOKEN x vocab_size

            B, TEXT_TOKEN, vocab_size = batch_text_logits.size()

            ############################################################################
            target_text_tokens = torch.argmax(
                target_text_mask_tensor, dim=1, keepdim=False
            )
            batch_text_loss = torch.tensor(0.0, device=batch_target_text_token.device)
            for bi, token in zip(
                torch.arange(B, device=batch_target_text_token.device),
                target_text_tokens,
            ):
                target_text_logits = batch_text_logits[bi][: token + 1]
                target_text_token = batch_target_text_token[bi][: token + 1]
                target_text_loss = F.cross_entropy(
                    target_text_logits, target_text_token, reduction="mean"
                )
                batch_text_loss += target_text_loss

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


class ImgLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.img_embedding = ImageEmbedding(config=config)
        self.img_transfomer = ImgTransformer(config=config)
        self.img_flatten = nn.Flatten(start_dim=1)
        # IMG Feature Prj MLP
        self.img_linear1 = nn.Linear(
            in_features=config.img_patch_embedding,
            out_features=config.img_text_proj_features_layer1,
        )
        self.img_linear2 = nn.Linear(
            in_features=config.img_text_proj_features_layer1,
            out_features=config.img_text_proj_features_layer2,
        )
        self.img_proj = nn.Sequential(
            self.img_linear1,
            nn.GELU(),
            self.img_linear2,
            nn.LayerNorm(config.img_text_proj_features_layer2),
            nn.Dropout(config.img_text_proj_dropout),
        )
        self.img_softmax = nn.LogSoftmax(dim=-1)
        self.img_norm = nn.LayerNorm(config.img_patch_embedding)
        self.img_token_weight = nn.Parameter(torch.rand(config.img_patches))

        # self.text_embedding = TextTokenEmbedding(config=config)
        self.text_transformer = TextMaskedTransformer(config=config)
        self.text_flatten = nn.Flatten(start_dim=1)
        self.text_norm = nn.LayerNorm(config.text_token_embedding)

        # Text Feature Prj MLP
        self.text_linear1 = nn.Linear(
            in_features=config.text_token_embedding,
            out_features=config.img_text_proj_features_layer1,
        )
        self.text_linear2 = nn.Linear(
            in_features=config.img_text_proj_features_layer1,
            out_features=config.img_text_proj_features_layer2,
        )
        self.text_proj = nn.Sequential(
            self.text_linear1,
            nn.GELU(),
            self.text_linear2,
            nn.LayerNorm(config.img_text_proj_features_layer2),
            nn.Dropout(config.img_text_proj_dropout),
        )
        self.text_softmax = nn.LogSoftmax(dim=-1)

        self.diag_mask = torch.diag(torch.ones(config.img_text_proj_features_layer2))
        self.loss_fn = nn.NLLLoss()
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.bos_token = self.text_transformer.text_token_embedding.text_encoder.encode(
            "<bos>"
        )[
            1
        ]  # return <bos><bos>
        self.img_caption_model = ImgCaptionModel(
            config=config,
            tokenizer=self.text_transformer.text_token_embedding.text_encoder,
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.img_linear1.weight, std=self.img_linear1.in_features**-0.5)
        nn.init.normal_(self.img_linear2.weight, std=self.img_linear2.in_features**-0.5)
        nn.init.normal_(
            self.text_linear1.weight, std=self.text_linear1.in_features**-0.5
        )
        nn.init.normal_(
            self.text_linear2.weight, std=self.text_linear2.in_features**-0.5
        )
        nn.init.normal_(self.img_token_weight, std=0.2)

    def forward(
        self,
        batch_aug_img_tensor1: torch.tensor,
        batch_aug_img_tensor2: torch.tensor,
        batch_text_tensor: torch.tensor,
        batch_text_mask_tensor: torch.tensor,
        batch_img_id_tensor: torch.tensor = None,
    ):
        """
        batch_aug_img_tensor1: B x IMG_PATCHES x IMG_EMB
        batch_aug_img_tensor2: B x IMG_PATCHES x IMG_EMB
        batch_text_tensor: B x TEXT_TOKEN
        batch_text_mask_tensor: B x TEXT_TOKEN
        """

        def _get_batch_img_feature(
            batch_aug_img_tensor: torch.tensor,
        ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
            img_embedding = self.img_embedding(
                batch_aug_img_tensor
            )  # B x IMG_PATCHES x IMG_EMB
            # print(f"img_encoding: {img_embedding.size()}")

            img_feature = self.img_transfomer(
                img_embedding
            )  # B x IMG_PATCHES x IMG_EMB
            img_feature = self.img_norm(img_feature)  # B x IMG_PATCHES x IMG_EMB
            img_contrastive_feature = self.img_norm(
                img_feature[:, -1, :]
            )  # B x IMG_EMB, take the last one
            # img_contrastive_feature = self.img_norm(
            #     torch.einsum(
            #         "bnf,n->bf",
            #         img_feature,
            #         self.img_token_weight.to(batch_img_tensor.device),
            #     )
            # )  # B x IMG_EMB
            # print(f"img_feature: {img_feature.size()}")

            # img_feature_flatten = self.img_flatten(img_feature)
            # print(f"img_feature_flatten: {img_feature_flatten.size()}")

            img_feature_proj = self.img_proj(img_contrastive_feature)
            # print(f"img_feature_proj: {img_feature_proj.size()}")  # B x img_text_proj_features
            return img_feature, img_contrastive_feature, img_feature_proj

        img_feature1, img_contrastive_feature1, img_feature_proj1 = (
            _get_batch_img_feature(batch_aug_img_tensor=batch_aug_img_tensor1)
        )

        img_feature2, img_contrastive_feature2, img_feature_proj2 = (
            _get_batch_img_feature(batch_aug_img_tensor=batch_aug_img_tensor2)
        )
        img_img_contrastive_scores = img_feature_proj1 @ img_feature_proj2.T
        img_img_ontrastive_prob = self.img_softmax(img_img_contrastive_scores)
        target = torch.arange(
            img_img_ontrastive_prob.size()[0], device=img_img_ontrastive_prob.device
        )
        img_img_loss = self.loss_fn(img_img_ontrastive_prob, target)

        # text_embedding = self.text_embedding(batch_text_tensor)
        # print(f"text_embedding: {text_embedding.size()}")

        text_feature = self.text_transformer(batch_text_tensor)
        text_feature = self.text_norm(text_feature)
        text_contrastive_feature = self.text_norm(
            text_feature[
                torch.arange(text_feature.shape[0]),
                torch.argmax(batch_text_mask_tensor, dim=1, keepdim=False),
            ]
        )
        # print(f"text_feature: {text_feature.size()}")

        # text_feature_flatten = self.text_flatten(text_feature)
        # print(f"text_feature_flatten: {text_feature_flatten.size()}")

        text_feature_proj = self.text_proj(text_contrastive_feature)
        # print(f"text_feature_proj: {text_feature_proj.size()}")  # B x img_text_proj_features

        # Contrastive learning
        img_text_contrastive_scores = img_feature_proj1 @ text_feature_proj.T
        # print(f"contractive_scores: {contrastive_scores}")  # B x img_text_proj_features

        # img_loss = constrastive_logit_loss(contrastive_scores)
        # text_loss = constrastive_logit_loss(contrastive_scores.T)

        img_text_contrastive_prob = self.img_softmax(img_text_contrastive_scores)
        # print(f"img_contrastive_prob: {img_contrastive_prob}")  # B x img_text_proj_features

        # ===============================================================================
        # Img BCE-Loss
        # ===============================================================================
        # target = torch.eye(
        #     contrastive_scores.size()[0], device=contrastive_scores.device
        # )
        # img_loss = self.bce_loss_fn(contrastive_scores, target)
        # text_loss = self.bce_loss_fn(contrastive_scores.T, target)

        # ===============================================================================
        # Img NLL-Loss
        # ===============================================================================
        target = torch.arange(
            img_text_contrastive_prob.size()[0], device=img_text_contrastive_prob.device
        )
        img_text_loss = self.loss_fn(img_text_contrastive_prob, target)
        # img_loss = self.loss_fn(img_contrastive_prob, self.target.expand(img_contrastive_prob.size()[0], -1))
        # print(f"img_loss: {img_loss}")

        text_contrastive_prob = self.text_softmax(img_text_contrastive_scores.T)

        # ===============================================================================
        # Text NLL-Loss
        # ===============================================================================
        # print(f"text_contrastive_prob: {text_contrastive_prob}")  # B x img_text_proj_features
        text_img_loss = self.loss_fn(text_contrastive_prob, target)
        # print(f"text_img_loss: {text_img_loss}")

        bos_embedding = self.text_transformer.text_token_embedding(
            x=torch.tensor(self.bos_token, device=batch_aug_img_tensor1.device),
            skip_position_embedding=True,
        )
        lm_logits, lm_loss = self.img_caption_model(
            img_feature=img_feature1,
            text_feature=text_feature,
            text_mask=(batch_text_tensor != 0),
            batch_target_text_token=batch_text_tensor,
            target_text_mask_tensor=batch_text_mask_tensor,
            bos_embedding=bos_embedding,
        )
        # print(f"lm_logits: {lm_logits.size()}")
        # print(f"lm_loss: {lm_loss}")

        return (
            img_img_loss,
            img_text_loss,
            text_img_loss,
            img_img_ontrastive_prob,
            img_text_contrastive_prob,
            text_contrastive_prob,
            lm_loss,
            lm_logits,
        )
