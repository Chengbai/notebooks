import torch
import torch.nn as nn
import pandas as pd
import os

os.environ["TOKENIZERS_PARALLELISM"] = "1"

from config import Config
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


class FlikerCommentTokenizer:
    # TOKENS = 128_000
    TOKENS = 10_000

    @staticmethod
    def train_tokenizer(config: Config):
        from fliker_img_comment_dataset import ImgCommentDataset

        def _batch_iterator(comments, batch_size=1000):
            for i in tqdm(range(0, len(comments), batch_size)):
                yield comments[i : i + batch_size]

        # Special setup to make train split contains all of the data, in turn we can get all of the comments.
        dataset = ImgCommentDataset(
            config=config,
            split="train",
            split_portions=(1.0, 0, 0),
        )
        assert dataset.img_comments_df is not None
        assert "comment" in dataset.img_comments_df
        comments = list(dataset.img_comments_df["comment"])

        cache_dir = f"/tmp/{config.hf_tokenizer_model_id}"
        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_tokenizer_model_id,
            token=config.hf_access_token,
            force_download=False,
            cache_dir=cache_dir,
            padding_side="right",
        )

        new_tokenizer = tokenizer.train_new_from_iterator(
            _batch_iterator(comments=comments),
            vocab_size=FlikerCommentTokenizer.TOKENS,
        )

        new_tokenizer.save_pretrained(config.fliker_comment_tokenizer_local_path)
        print(
            f"Saved new fliker comment tokenizer at: {config.fliker_comment_tokenizer_local_path}"
        )
        assert len(new_tokenizer) == FlikerCommentTokenizer.TOKENS
        return new_tokenizer

    @staticmethod
    def get_tokenizer(config: Config):
        if Path(config.fliker_comment_tokenizer_local_path).is_dir():
            cache_dir = f"/tmp/{config.hf_tokenizer_model_id}"
            tokenizer = AutoTokenizer.from_pretrained(
                config.fliker_comment_tokenizer_local_path,
                token=config.hf_access_token,
                force_download=False,
                cache_dir=cache_dir,
                padding_side="right",
            )
        else:
            tokenizer = FlikerCommentTokenizer.train_tokenizer(config=config)

        print(f"tokens: {len(tokenizer)}")
        print(f"tokenizer.is_fast: {tokenizer.is_fast}")
        return tokenizer
