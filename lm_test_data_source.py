from config import Config
from image_comment_data_item import ImgCommentDataItem
from pathlib import Path
from typing import List


def get_test_data_items() -> List[ImgCommentDataItem]:
    config = Config()
    fliker_imgs_folder = Path(config.fliker_img_comments_folder) / "flickr30k_images"

    return [
        ImgCommentDataItem(
            image_name=fliker_imgs_folder / "3212671393.jpg",
            comment="",
            comment_number=0,
            image_id=0,
            source="lm_test",
        ),
        ImgCommentDataItem(
            image_name=fliker_imgs_folder / "3273585735.jpg",
            comment="",
            comment_number=1,
            image_id=1,
            source="lm_test",
        ),
    ]
