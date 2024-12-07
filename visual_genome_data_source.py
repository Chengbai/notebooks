import json

from config import Config
from common_util import load_json
from dataclasses import dataclass
from image_comment_data_item import ImgCommentDataItem
from pathlib import Path
from typing import Dict, List


@dataclass
class VisualGenomeImgInfo:
    image: str
    caption: str
    image_id: str  # e.g "vg_1"

    def to_data_item(self) -> ImgCommentDataItem:
        item_id = int(self.image_id.replace("vg_", ""))
        return ImgCommentDataItem(
            source="visual_genome",
            image_name=self.image,
            image_id=item_id,
            comment=self.caption,
            comment_number=item_id,  # not unique, one image could have multiple captions. Use the same `image_id` to connect them.
        )


def load_visual_genome_data_items(config: Config) -> List[ImgCommentDataItem]:
    assert config is not None

    visual_genome_train_file = (
        Path(config.visual_genome_img_comments_folder) / "vg.json"
    )
    captions_train_json = load_json(visual_genome_train_file)
    visual_genome_img_info = [
        VisualGenomeImgInfo(**img_annotation) for img_annotation in captions_train_json
    ]

    # Reset the img file path
    for img_info in visual_genome_img_info:
        img_file_path = (
            Path(config.visual_genome_img_comments_folder) / Path(img_info.image).name
        )
        img_info.image = str(img_file_path) if img_file_path.exists() else None
        # Reset to None if the image file is not exist

    # Filer out if the image file is missing
    data_items = [
        annotation.to_data_item()
        for annotation in visual_genome_img_info
        if annotation.image is not None
    ]
    return data_items
