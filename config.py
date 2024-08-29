import os
from pathlib import Path


class Config:
    # fliker 30k img comment file path
    img_comments_folder = Path("/Users/chengbai/ml/dataset/flickr30k_images/")

    # Image
    img_patch_size = 16

    img_w_size = img_patch_size * 32  # 512
    img_h_size = img_patch_size * 32  # 512

    img_patches = (img_w_size // img_patch_size) * (img_h_size // img_patch_size)
    img_patch_embedding = 728

    # Img Transform
    img_hidden = 1024
    img_transformer_heads = 8
    img_dropout = 0.0
    img_transformer_blocks = 6

    # Text
    text_tiktokenizer = "o200k_base"
    max_text_len = 50
    text_token_embedding = 728
    text_transformer_heads = 8
    text_transformer_blocks = 6
    text_dropout = 0.0

    # Construstrive Learning
    img_text_proj_features = 1024

    # huggingface
    # Create access token via: https://huggingface.co/settings/tokens, and add it into the env variable `hf_access_token`
    hf_access_token = os.environ["hf_access_token"]
    hf_tokenizer_model_id = "google/paligemma-3b-mix-224"
    fliker_comment_tokenizer_local_path = (
        Path(os.path.dirname(__file__)) / "paligemma-3b-mix-224-tokenizer"
    )
