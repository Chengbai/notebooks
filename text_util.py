import torch

from config import Config
from typing import List, Tuple


def normalize_comment(
    config: Config, comment_tokens: List[int]
) -> Tuple[torch.tensor, torch.tensor]:
    if len(comment_tokens) > config.max_text_len:
        comment_tokens = comment_tokens[: config.max_text_len]
        comment_mask = torch.tensor([1] * config.max_text_len, dtype=torch.int8)
    else:
        comment_mask = torch.concat(
            [
                torch.arange(
                    start=1,
                    end=len(comment_tokens) + 1,
                    step=1,
                    dtype=torch.int8,
                ),
                torch.tensor(
                    [0] * (config.max_text_len - len(comment_tokens)),
                    dtype=torch.int8,
                ),
            ]
        )

        # TODO: review append `<pad>` - 0 logic
        comment_tokens = comment_tokens + [
            0 for _ in range(config.max_text_len - len(comment_tokens))
        ]
    comment_tokens = torch.tensor(comment_tokens, dtype=torch.long)
    return comment_tokens, comment_mask
