import tiktoken
import pandas as pd

from common_util import get_logger
from config import Config
from fliker_comment_tokenizer import FlikerCommentTokenizer
from image_comment_data_item import ImgCommentDataItem
from fliker_data_source import load_fliker_data_items
from coco_data_source import load_coco_data_items
from visual_genome_data_source import load_visual_genome_data_items
from img_util import load_img_tensor
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from text_util import normalize_comment

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple

COMMENT = "comment"
COMMENT_NUMBER = "comment_number"
IMAGE_ID = "image_id"
IMAGE_NAME = "image_name"
TRAIN = "train"
EVAL = "eval"
TEST = "test"

logger = get_logger(__name__)


# Create Dataset
class ImgCommentDataset(Dataset):
    def __init__(
        self,
        config: Config,
        split: str = TRAIN,
        split_portions: Tuple[float, float] = (0.72, 0.18, 0.1),
    ):
        self.config = config
        fliker_data_items: List[ImgCommentDataItem] = load_fliker_data_items(
            config=config
        )
        logger.info(f"Loaded {len(fliker_data_items)} fliker image-caption data items.")

        coco_data_items: List[ImgCommentDataItem] = load_coco_data_items(config=config)
        logger.info(f"Loaded {len(coco_data_items)} coco image-caption data items.")

        visaul_genome_data_items: List[ImgCommentDataItem] = (
            load_visual_genome_data_items(config=config)
        )
        logger.info(
            f"Loaded {len(visaul_genome_data_items)} visual-genome image-caption data items."
        )

        total_data_items = (
            fliker_data_items + coco_data_items + visaul_genome_data_items
        )
        df = pd.DataFrame([data_item.to_json() for data_item in total_data_items])

        self.split = split
        self.split_portions = split_portions

        train_split_index = int(len(df) * self.split_portions[0])
        eval_split_index = int(
            len(df) * (self.split_portions[0] + self.split_portions[1])
        )
        if self.split == TRAIN:
            self.img_comments_df = df[:train_split_index]
        elif self.split == EVAL:
            self.img_comments_df = df[train_split_index:eval_split_index]
        else:
            assert self.split == TEST
            self.img_comments_df = df[eval_split_index:]

        # self.text_tokenizer = tiktoken.get_encoding(config.text_tiktokenizer)
        self.text_tokenizer = None

    def _get_img_cache_file(self, idx: int) -> Path:
        folder_path = self.config.fliker_img_comments_folder / "cache" / self.split
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / f"img_tensor_{idx}.pt"

    def _get_comment_cache_file(self, idx: int) -> Path:
        folder_path = self.config.fliker_img_comments_folder / "cache" / self.split
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / f"comment_tokens_{idx}.pt"

    def _get_comment_mask_cache_file(self, idx: int) -> Path:
        folder_path = self.config.fliker_img_comments_folder / "cache" / self.split
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / f"comment_mask_{idx}.pt"

    def __len__(self):
        return len(self.img_comments_df)

    def __getitem__(self, idx: int):
        # logger.info(f"idx: {idx}")
        idx = idx % len(self.img_comments_df)
        # check cache first
        if (
            False
            and self._get_img_cache_file(idx).is_file()
            and self._get_comment_cache_file(idx).is_file()
        ):
            img_tensor = torch.load(self._get_img_cache_file(idx))
            comment_tokens = torch.load(self._get_comment_cache_file(idx))
            comment_ask = torch.load(self._get_comment_mask_cache_file(idx))
            item = self.img_comments_df.iloc[idx]
            img_id = torch.tensor(item[IMAGE_ID], dtype=torch.int)
        else:
            item = self.img_comments_df.iloc[idx]
            image_name = item[IMAGE_NAME]
            img_id = item[IMAGE_ID]
            comment_number = item[COMMENT_NUMBER]
            comment = str(item[COMMENT])
            if not comment:
                logger.info(f"missing comment for image: {image_name}")

            # row_df = self.img_comments_df[idx : idx + 1]
            # image_name = str(list(row_df[IMAGE_NAME])[0])
            assert Path(image_name).is_file(), f"cannot find file: {image_name}"
            img_id = torch.tensor(img_id, dtype=torch.int)

            # FlikerCommentTokenizer `encode` always auto prefix with `<bos>`
            if self.text_tokenizer is None:
                self.text_tokenizer = FlikerCommentTokenizer.get_tokenizer(
                    config=self.config
                )
            comment_tokens = self.text_tokenizer.encode(comment)

            comment_tokens, comment_mask = normalize_comment(
                config=self.config, comment_tokens=comment_tokens
            )
            assert len(comment_tokens) == self.config.max_text_len

            # return load_img_tensor(image_name), comment_number, comment, comment_tokens
            img_aug_tensor1, img_aug_tensor2 = load_img_tensor(self.config, image_name)

        return (
            img_aug_tensor1,
            img_aug_tensor2,
            img_id,
            comment_tokens,
            comment_mask,
        )

    def cache_data(self):
        for idx in tqdm(range(len(self)), total=len(self)):
            img_tensor, img_id, comment_tokens, comment_mask = self[idx]
            torch.save(
                img_tensor,
                self._get_img_cache_file(idx),
            )
            torch.save(
                comment_tokens,
                self._get_comment_cache_file(idx),
            )
            torch.save(
                comment_mask,
                self._get_comment_mask_cache_file(idx),
            )
