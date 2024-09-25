import unittest
import torch

from config import Config
from typing import List, Tuple
from text_util import normalize_comment

config = Config()


class TestNormalizeComment(unittest.TestCase):

    def test_empty(self):
        comment_tokens, comment_mask = normalize_comment(
            config=config, comment_tokens=[]
        )
        v, i = torch.max(comment_mask[None, :], dim=1, keepdim=False)
        self.assertEqual(v, 0)
        self.assertEqual(i, 0)
        self.assertIsNotNone(comment_tokens)
        self.assertIsNotNone(comment_mask)

    def test_too_long_comment_tokens(self):
        too_long_comment_tokens = torch.arange(
            config.max_text_len + 1, dtype=torch.long
        )
        comment_tokens, comment_mask = normalize_comment(
            config=config, comment_tokens=too_long_comment_tokens
        )
        self.assertTrue(
            torch.equal(too_long_comment_tokens[: config.max_text_len], comment_tokens)
        )
        self.assertTrue(
            torch.equal(
                torch.arange(
                    start=1, end=config.max_text_len + 1, step=1, dtype=torch.long
                ),
                comment_mask,
            )
        )


if __name__ == "__main__":
    unittest.main()
