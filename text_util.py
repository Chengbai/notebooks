import torch

from config import Config
from typing import List, Tuple


def normalize_comment(
    config: Config, comment_tokens: List[int]
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Input `comment_tokens` could vary length case by case.
    After normalization, output:
     - all the comment tokens will have same length as `config.max_text_len`
     - a mask is return with pattern: [1, 2, ..., N, 0, 0, ...]. The `MAX value` is the lengh of the TRUE token length
    """
    if len(comment_tokens) > config.max_text_len:
        comment_tokens = comment_tokens[: config.max_text_len]
        comment_mask = torch.arange(
            start=1,
            end=config.max_text_len + 1,
            step=1,
            dtype=torch.long,
        )
    else:
        comment_mask = torch.concat(
            [
                torch.arange(
                    start=1,
                    end=len(comment_tokens) + 1,
                    step=1,
                    dtype=torch.long,
                ),
                torch.tensor(
                    [0] * (config.max_text_len - len(comment_tokens)),
                    dtype=torch.long,
                ),
            ]
        )

        # TODO: review append `<pad>` - 0 logic
        comment_tokens = comment_tokens + [
            0 for _ in range(config.max_text_len - len(comment_tokens))
        ]
    comment_tokens = torch.tensor(comment_tokens, dtype=torch.long)
    return comment_tokens, comment_mask
