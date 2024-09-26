import os
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    # fliker 30k img comment file path
    fliker_img_comments_folder = Path("/Users/chengbai/ml/dataset/flickr30k_images/")
    coco_img_comments_folder = Path("/Users/chengbai/ml/dataset/coco/")

    # Image
    img_patch_size = 16

    img_w_size = img_patch_size * 14  # 224
    img_h_size = img_patch_size * 14  # 224

    img_patches = (img_w_size // img_patch_size) * (img_h_size // img_patch_size)
    img_patch_embedding = 512

    # Img Transform
    img_hidden = 1024
    img_transformer_heads = 8
    img_dropout = 0.0
    img_transformer_blocks = 12

    # Text
    text_tiktokenizer = "o200k_base"
    max_text_len = 50
    text_token_embedding = 512
    text_transformer_heads = 8
    text_transformer_blocks = 12
    text_dropout = 0.0

    # Construstrive Learning
    img_text_proj_features_layer1 = 512
    img_text_proj_features_layer2 = 256
    img_text_proj_dropout = 0.0

    # huggingface
    # Create access token via: https://huggingface.co/settings/tokens, and add it into the env variable `hf_access_token`
    hf_access_token = os.environ["hf_access_token"]
    hf_tokenizer_model_id = "google/paligemma-3b-mix-224"
    fliker_comment_tokenizer_local_path = (
        Path(os.path.dirname(__file__)) / "paligemma-3b-mix-224-tokenizer"
    )

    # loss
    img_img_loss_weight = 2.0
    img_text_loss_weight = 2.0
    text_img_loss_weight = 2.0
    lm_loss_weight = 1.0

    # rolling window cache recent features
    rolling_cache_enabled = True
    rolling_cache_warm_steps = 1000
    rolling_cache_size = 10
