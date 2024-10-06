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
from text_casual_mask_transformer import TextMaskedTransformer
from tqdm import tqdm
from typing import Tuple
from vlm_img_caption_model import ImgCaptionModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as VF


logger = get_logger(__name__)


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

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # https://paperswithcode.com/method/weight-tying
        self.img_caption_model.transformer.text_token_embedding.embeddings.weight = (
            self.text_transformer.text_token_embedding.embeddings.weight
        )

        self.rolling_cache = {}

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

    def get_batch_img_feature(
        self,
        batch_aug_img_tensor: torch.tensor,
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        img_embedding = self.img_embedding(
            batch_aug_img_tensor
        )  # B x IMG_PATCHES x IMG_EMB
        # logger.info(f"img_encoding: {img_embedding.size()}")

        img_feature = self.img_transfomer(img_embedding)  # B x IMG_PATCHES x IMG_EMB
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
        # logger.info(f"img_feature: {img_feature.size()}")

        # img_feature_flatten = self.img_flatten(img_feature)
        # logger.info(f"img_feature_flatten: {img_feature_flatten.size()}")

        img_feature_proj = self.img_proj(img_contrastive_feature)
        # logger.info(f"img_feature_proj: {img_feature_proj.size()}")  # B x img_text_proj_features
        return img_feature, img_contrastive_feature, img_feature_proj

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
        img_feature1, img_contrastive_feature1, img_feature_proj1 = (
            self.get_batch_img_feature(batch_aug_img_tensor=batch_aug_img_tensor1)
        )

        img_feature2, img_contrastive_feature2, img_feature_proj2 = (
            self.get_batch_img_feature(batch_aug_img_tensor=batch_aug_img_tensor2)
        )

        cached_img_feature_proj1 = self.rolling_cache.get("img_feature_proj1", None)
        cached_img_feature_proj2 = self.rolling_cache.get("img_feature_proj2", None)

        if (
            self.config.rolling_cache_enabled
            and cached_img_feature_proj1 is not None
            and cached_img_feature_proj2 is not None
        ):
            assert cached_img_feature_proj1.size()[0] <= self.config.rolling_cache_size
            assert cached_img_feature_proj2.size()[0] <= self.config.rolling_cache_size
            all_img_feature_proj1 = torch.vstack(
                [
                    img_feature_proj1,
                    cached_img_feature_proj1.to(device=batch_aug_img_tensor1.device),
                ],
            )
            all_img_feature_proj2 = torch.vstack(
                [
                    img_feature_proj2,
                    cached_img_feature_proj2.to(device=batch_aug_img_tensor2.device),
                ],
            )
        else:
            all_img_feature_proj1 = img_feature_proj1
            all_img_feature_proj2 = img_feature_proj2

        img_img_contrastive_scores = (
            all_img_feature_proj1 @ all_img_feature_proj2.T
        )  # (B+CACHE) x IMG_PRJ @ IMG_PRJ x (B+CACHE) => (B+CACHE) x (B+CACHE)
        img_img_contrastive_prob = self.img_softmax(img_img_contrastive_scores)
        target = torch.arange(
            img_img_contrastive_prob.size()[0], device=img_img_contrastive_prob.device
        )
        img_img_loss = self.loss_fn(img_img_contrastive_prob, target)

        # text_embedding = self.text_embedding(batch_text_tensor)
        # logger.info(f"text_embedding: {text_embedding.size()}")

        text_feature = self.text_transformer(batch_text_tensor)
        text_feature = self.text_norm(text_feature)
        text_contrastive_feature = self.text_norm(
            text_feature[
                torch.arange(text_feature.shape[0]),
                torch.argmax(batch_text_mask_tensor, dim=1, keepdim=False),
            ]
        )
        # logger.info(f"text_feature: {text_feature.size()}")

        # text_feature_flatten = self.text_flatten(text_feature)
        # logger.info(f"text_feature_flatten: {text_feature_flatten.size()}")

        text_feature_proj = self.text_proj(text_contrastive_feature)
        # logger.info(f"text_feature_proj: {text_feature_proj.size()}")  # B x img_text_proj_features

        cached_text_feature_proj = self.rolling_cache.get("text_feature_proj", None)
        if self.config.rolling_cache_enabled and cached_text_feature_proj is not None:
            assert cached_img_feature_proj1 is not None
            assert cached_text_feature_proj.size() == cached_img_feature_proj1.size()

            all_text_feature_proj = torch.vstack(
                [
                    text_feature_proj,
                    cached_text_feature_proj.to(device=batch_text_tensor.device),
                ],
            )
        else:
            all_text_feature_proj = text_feature_proj

        # Contrastive learning
        img_text_contrastive_scores = all_img_feature_proj1 @ all_text_feature_proj.T
        # logger.info(f"contractive_scores: {contrastive_scores}")  # B x img_text_proj_features

        # img_loss = constrastive_logit_loss(contrastive_scores)
        # text_loss = constrastive_logit_loss(contrastive_scores.T)

        img_text_contrastive_prob = self.img_softmax(img_text_contrastive_scores)
        # logger.info(f"img_contrastive_prob: {img_contrastive_prob}")  # B x img_text_proj_features

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
        # logger.info(f"img_loss: {img_loss}")

        text_contrastive_prob = self.text_softmax(img_text_contrastive_scores.T)

        # ===============================================================================
        # Text NLL-Loss
        # ===============================================================================
        # logger.info(f"text_contrastive_prob: {text_contrastive_prob}")  # B x img_text_proj_features
        text_img_loss = self.loss_fn(text_contrastive_prob, target)
        # logger.info(f"text_img_loss: {text_img_loss}")

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
        # logger.info(f"lm_logits: {lm_logits.size()}")
        # logger.info(f"lm_loss: {lm_loss}")

        # Manage the cache
        if self.config.rolling_cache_enabled:
            all_img_feature_proj1 = (
                all_img_feature_proj1[: self.config.rolling_cache_size].clone().detach()
            )
            all_img_feature_proj2 = (
                all_img_feature_proj2[: self.config.rolling_cache_size].clone().detach()
            )
            all_text_feature_proj = (
                all_text_feature_proj[: self.config.rolling_cache_size].clone().detach()
            )
            self.rolling_cache["img_feature_proj1"] = all_img_feature_proj1
            self.rolling_cache["img_feature_proj2"] = all_img_feature_proj2
            self.rolling_cache["text_feature_proj"] = all_text_feature_proj

        # logger.info(
        #     f'self.rolling_cache["text_feature_proj"]: {len(self.rolling_cache["text_feature_proj"])}, self.rolling_cache["img_feature_proj1"]: {len(self.rolling_cache["img_feature_proj1"])}, self.rolling_cache["img_feature_proj2"]: {len(self.rolling_cache["img_feature_proj2"])}'
        # )

        return (
            img_img_loss,
            img_text_loss,
            text_img_loss,
            img_img_contrastive_prob,
            img_text_contrastive_prob,
            text_contrastive_prob,
            lm_loss,
            lm_logits,
        )
