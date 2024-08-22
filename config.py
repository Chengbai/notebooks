class Config:
    # Image
    img_patch_size = 16

    img_w_size = img_patch_size * 32
    img_h_size = img_patch_size * 32

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
