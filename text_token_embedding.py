import torch
import torch.nn as nn

from config import Config
import tiktoken


class TextTokenEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.text_encoder = tiktoken.get_encoding(config.text_tiktokenizer)
        self.embeddings = nn.Embedding(
            self.text_encoder.n_vocab, config.text_token_embedding
        )

    def forward(self, x: torch.tensor):
        """
        x: B x token_idx
        ret: B x TEXT_TOKEN_EMB
        """
        return self.embeddings(x)
