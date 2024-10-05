import pandas as pd


from common_util import get_logger
from config import Config
from fliker_comment_tokenizer import FlikerCommentTokenizer
from img_util import load_img_tensor
from lm_test_data_source import get_test_data_items
from pathlib import Path
from text_util import normalize_comment


import torch
from torch.utils.data import Dataset, DataLoader


COMMENT = "comment"
COMMENT_NUMBER = "comment_number"
IMAGE_ID = "image_id"
IMAGE_NAME = "image_name"


logger = get_logger(__name__)


class LMTestDataset(Dataset):
    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.data_items = get_test_data_items()
        self.img_comments_df = pd.DataFrame(
            [data_item.to_json() for data_item in self.data_items]
        )
        self.text_tokenizer = FlikerCommentTokenizer.get_tokenizer(config=self.config)

    def __len__(self):
        return len(self.img_comments_df)

    def __getitem__(self, idx: int):
        # logger.info(f"idx: {idx}")
        idx = idx % len(self.img_comments_df)

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

    def reset_comment(self):
        for data_item in self.data_items:
            data_item.comment = ""
        self.img_comments_df = pd.DataFrame(
            [data_item.to_json() for data_item in self.data_items]
        )

    def update_comment(self, img_id: int, comment: str):
        for data_item in self.data_items:
            if data_item.image_id == img_id:
                data_item.comment = f"{data_item.comment}{comment}"
                break
        self.img_comments_df = pd.DataFrame(
            [data_item.to_json() for data_item in self.data_items]
        )

    def get_img_comment(self):
        return [
            {"image": data_item.image_name, "comment": data_item.comment}
            for data_item in self.data_items
        ]
