import tiktoken
import pandas as pd

from config import Config
from fliker_comment_tokenizer import FlikerCommentTokenizer
from image_comment_data_item import ImgCommentDataItem
from img_util import load_img_tensor
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List

IMAGE_NAME = "image_name"
IMAGE_ID = "image_id"


def enrich_fliker_data_items(
    config: Config, data_items: List[ImgCommentDataItem]
) -> List[ImgCommentDataItem]:
    """
    1. enrich the image_id. Each fliker image has 5 comments
    2. enrich image_name as the full file path
    """
    imgs_folder = Path(config.fliker_img_comments_folder) / "flickr30k_images"

    data_items = sorted(data_items, key=lambda r: r.image_name)
    img_id = 0
    prev_img_name = data_items[0].image_name
    for item in data_items:
        if item.image_name != prev_img_name:
            prev_img_name = item.image_name
            img_id += 1
        item.image_id = img_id

        item.image_name = imgs_folder / Path(item.image_name).name

    return data_items


def load_fliker_data_items(config: Config) -> List[ImgCommentDataItem]:
    img_comments_file = Path(config.fliker_img_comments_folder) / "results.csv"

    # The current `results.csv` file is using "| " to seperate 3 columns.
    # For the pd.read_csv, the `sep` here is given as a regular expression.
    df = pd.read_csv(img_comments_file, sep="|")
    df["source"] = "fliker30k"
    data_items = [ImgCommentDataItem(**record) for record in df.to_dict("records")]

    # In Fliker data, one image have 5 comments
    data_items = enrich_fliker_data_items(config=config, data_items=data_items)

    return data_items
