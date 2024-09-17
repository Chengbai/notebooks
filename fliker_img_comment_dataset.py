import tiktoken
import pandas as pd

from config import Config
from fliker_comment_tokenizer import FlikerCommentTokenizer
from img_util import load_img_tensor
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple

COMMENT = "comment"
COMMENT_NUMBER = "comment_number"
IMAGE_ID = "image_id"
IMAGE_NAME = "image_name"
TRAIN = "train"
EVAL = "eval"
TEST = "test"


def enrich_img_id(df: pd.DataFrame) -> pd.DataFrame:
    assert IMAGE_NAME in df.columns
    df_records: list[dict] = df.to_dict("records")
    df_records = sorted(df_records, key=lambda r: r[IMAGE_NAME])
    img_id = 0
    prev_img_name = df_records[0][IMAGE_NAME]
    for record in df_records:
        if record[IMAGE_NAME] != prev_img_name:
            prev_img_name = record[IMAGE_NAME]
            img_id += 1
        record[IMAGE_ID] = img_id

    enriched_df = pd.DataFrame(df_records)
    enriched_csv_file = "/tmp/enriched_results.csv"
    enriched_df.to_csv(enriched_csv_file, sep="|")
    print(f"Enriched img id: {enriched_csv_file}")
    return enriched_df


# Create Dataset
class ImgCommentDataset(Dataset):
    def __init__(
        self,
        config: Config,
        split: str = TRAIN,
        split_portions: Tuple[float, float] = (0.72, 0.18, 0.1),
    ):
        self.config = config
        self.img_comments_file = config.img_comments_folder / "results.csv"
        self.imgs_folder = config.img_comments_folder / "flickr30k_images"

        self.split = split
        self.split_portions = split_portions

        # The current `results.csv` file is using "| " to seperate 3 columns.
        # For the pd.read_csv, the `sep` here is given as a regular expression.
        df = pd.read_csv(self.img_comments_file, sep="|")
        df = enrich_img_id(df)
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
        folder_path = self.config.img_comments_folder / "cache" / self.split
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / f"img_tensor_{idx}.pt"

    def _get_comment_cache_file(self, idx: int) -> Path:
        folder_path = self.config.img_comments_folder / "cache" / self.split
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / f"comment_tokens_{idx}.pt"

    def _get_comment_mask_cache_file(self, idx: int) -> Path:
        folder_path = self.config.img_comments_folder / "cache" / self.split
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path / f"comment_mask_{idx}.pt"

    def __len__(self):
        return len(self.img_comments_df)

    def __getitem__(self, idx: int):
        # print(f"idx: {idx}")
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

            # row_df = self.img_comments_df[idx : idx + 1]
            # image_name = str(list(row_df[IMAGE_NAME])[0])
            assert (
                self.imgs_folder / image_name
            ).is_file(), f"cannot find file: {self.imgs_folder/image_name}"
            img_id = torch.tensor(img_id, dtype=torch.int)

            # FlikerCommentTokenizer `encode` always auto prefix with `<bos>`
            if self.text_tokenizer is None:
                self.text_tokenizer = FlikerCommentTokenizer.get_tokenizer(
                    config=self.config
                )
            comment_tokens = self.text_tokenizer.encode(comment)
            if len(comment_tokens) > self.config.max_text_len:
                comment_tokens = comment_tokens[: self.config.max_text_len]
                comment_mask = torch.tensor(
                    [1] * self.config.max_text_len, dtype=torch.int8
                )
            else:
                comment_mask = torch.concat(
                    [
                        torch.arange(
                            start=1,
                            end=len(comment_tokens) + 1,
                            step=1,
                            dtype=torch.int8,
                        ),
                        torch.tensor(
                            [0] * (self.config.max_text_len - len(comment_tokens)),
                            dtype=torch.int8,
                        ),
                    ]
                )

                # TODO: review append `<pad>` - 0 logic
                comment_tokens = comment_tokens + [
                    0 for _ in range(self.config.max_text_len - len(comment_tokens))
                ]

            assert len(comment_tokens) == self.config.max_text_len
            comment_tokens = torch.tensor(comment_tokens, dtype=torch.long)

            # return load_img_tensor(self.imgs_folder/image_name), comment_number, comment, comment_tokens
            img_aug_tensor1, img_aug_tensor2 = load_img_tensor(
                self.config, self.imgs_folder / image_name
            )

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
