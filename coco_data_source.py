import json

from config import Config
from common_util import load_json
from dataclasses import dataclass
from image_comment_data_item import ImgCommentDataItem
from pathlib import Path
from typing import Dict, List


@dataclass
class CocoAnnotation:
    image_id: int
    id: int
    caption: str
    image_file: str = None  # enriched

    def to_data_item(self) -> ImgCommentDataItem:
        return ImgCommentDataItem(
            source="coco",
            image_name=self.image_file,
            image_id=self.id,
            comment=self.caption,
            comment_number=self.id,
        )


@dataclass
class CocoImgInfo:
    license: int
    file_name: str
    coco_url: str
    height: int
    width: int
    flickr_url: str
    id: int
    date_captured: str


def load_coco_data_items(config: Config) -> List[ImgCommentDataItem]:
    assert config is not None

    coco_imgs_folder = (
        config.coco_img_comments_folder / "train2017"
    )  # /Users/chengbai/ml/dataset/flickr30k_images/flickr30k_images

    captions_train_file = (
        config.coco_img_comments_folder / "annotations_2017" / "captions_train2017.json"
    )  # /Users/chengbai/ml/dataset/flickr30k_images/results.csv

    captions_train_json = load_json(captions_train_file)

    coco_img_info = [
        CocoImgInfo(**img_annotation)
        for img_annotation in captions_train_json["images"]
    ]
    coco_img_id_to_info = {info.id: info for info in coco_img_info}

    coco_annotations = [
        CocoAnnotation(**annotation)
        for annotation in captions_train_json["annotations"]
    ]

    for annotation in coco_annotations:
        annotation.image_file = (
            f"{coco_imgs_folder}/{coco_img_id_to_info[annotation.image_id].file_name}"
        )

    data_items = [annotation.to_data_item() for annotation in coco_annotations]
    return data_items
