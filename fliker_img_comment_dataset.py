import tiktoken
import pandas as pd

from config import Config
from img_util import load_img_tensor
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

COMMENT = "comment"
COMMENT_NUMBER = "comment_number"
IMAGE_ID = "image_id"
IMAGE_NAME = "image_name"
TRAIN = "train"
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
        img_comments_folder: Path,
        train_test_split: str = TRAIN,
        train_test_split_portion: float = 0.8,
    ):
        self.config = config
        self.img_commments_folder = img_comments_folder
        self.img_comments_file = img_comments_folder / "results.csv"
        self.imgs_folder = img_comments_folder / "flickr30k_images"

        self.train_test_split = train_test_split
        self.train_test_split_portion = train_test_split_portion

        # The current `results.csv` file is using "| " to seperate 3 columns.
        # For the pd.read_csv, the `sep` here is given as a regular expression.
        df = pd.read_csv(self.img_comments_file, sep="|")
        df = enrich_img_id(df)
        train_split_len = int(len(df) * self.train_test_split_portion)
        if self.train_test_split == TRAIN:
            self.img_comments_df = df[:train_split_len]
        else:
            self.img_comments_df = df[train_split_len:]

        self.text_encoder = tiktoken.get_encoding(config.text_tiktokenizer)

    def __len__(self):
        return len(self.img_comments_df)

    def __getitem__(self, idx: int):
        print(f"idx: {idx}")
        row_df = self.img_comments_df[idx : idx + 1]
        image_name = str(list(row_df[IMAGE_NAME])[0])
        assert (
            self.imgs_folder / image_name
        ).is_file(), f"cannot find file: {self.img_commments_folder/image_name}"
        img_id = int(list(row_df[IMAGE_ID])[0])
        img_id = torch.tensor(img_id, dtype=torch.int)

        comment_number = int(list(row_df[COMMENT_NUMBER])[0])
        comment = str(list(row_df[COMMENT])[0])
        comment_encoding = self.text_encoder.encode(comment)
        if len(comment_encoding) > self.config.max_text_len:
            comment_encoding = comment_encoding[: self.config.max_text_len]
        else:
            # TODO: review append `<|endoftext|>` - 199999 logic
            comment_encoding = comment_encoding + [
                199999 for _ in range(self.config.max_text_len - len(comment_encoding))
            ]
        assert len(comment_encoding) == self.config.max_text_len
        comment_encoding = torch.tensor(comment_encoding, dtype=torch.int)

        # return load_img_tensor(self.imgs_folder/image_name), comment_number, comment, comment_encoding
        img_tensor = load_img_tensor(self.config, self.imgs_folder / image_name)

        return img_tensor, img_id, comment_encoding
