import os

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class Config:
    # fliker 30k img comment file path
    fliker_img_comments_folder: str = "/Users/chengbai/ml/dataset/flickr30k_images/"
    coco_img_comments_folder: str = "/Users/chengbai/ml/dataset/coco/"
    visual_genome_img_comments_folder: str = "/Users/chengbai/ml/dataset/visual_genome/"

    # Image
    img_patch_size: int = 16

    img_w_size: int = img_patch_size * 14  # 224
    img_h_size: int = img_patch_size * 14  # 224

    img_patches: int = (img_w_size // img_patch_size) * (img_h_size // img_patch_size)
    img_patch_embedding: int = 512

    # Img Transform
    img_hidden: int = 1024
    img_transformer_heads: int = 8
    img_dropout: float = 0.0
    img_transformer_blocks: int = 12

    # Text
    text_tiktokenizer: str = "o200k_base"
    max_text_len: int = 50
    text_token_embedding: int = 512
    text_transformer_heads: int = 8
    text_transformer_blocks: int = 12
    text_dropout: float = 0.0

    # Construstrive Learning
    img_text_proj_features_layer1: int = 512
    img_text_proj_features_layer2: int = 256
    img_text_proj_dropout: float = 0.0

    # huggingface
    # Create access token via: https://huggingface.co/settings/tokens, and add it into the env variable `hf_access_token`
    hf_access_token: str = os.environ["hf_access_token"]
    hf_tokenizer_model_id: str = "google/paligemma-3b-mix-224"
    fliker_comment_tokenizer_local_path: str = str(
        Path(os.path.dirname(__file__)) / "paligemma-3b-mix-224-tokenizer"
    )

    # loss
    img_img_loss_weight: float = 2.0
    img_text_loss_weight: float = 2.0
    text_img_loss_weight: float = 2.0
    lm_loss_weight: float = 1.0

    # rolling window cache recent features
    rolling_cache_enabled: bool = False
    rolling_cache_warm_steps: int = 1000
    rolling_cache_size: int = 1000

    # Image Caption
    sample_top_k: int = 10

    # Debug
    debugging: bool = False
    debugging_step_interval: int = 5000
    debugging_step_window: int = 10

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)
