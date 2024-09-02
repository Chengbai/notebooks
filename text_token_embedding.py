import torch
import torch.nn as nn

from config import Config
from fliker_comment_tokenizer import FlikerCommentTokenizer
import tiktoken


class TextTokenEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # self.text_encoder = tiktoken.get_encoding(config.text_tiktokenizer)
        self.text_encoder = FlikerCommentTokenizer.get_tokenizer(config=config)
        self.embeddings = nn.Embedding(
            self.text_encoder.vocab_size, config.text_token_embedding
        )
        self.pos_embedding = nn.Embedding(
            config.max_text_len, config.text_token_embedding
        )

    def forward(self, x: torch.tensor, skip_position_embedding: bool = False):
        """
        x: B x token_idx
        ret: B x TEXT_TOKEN_EMB
        """
        x_emb = self.embeddings(x)
        if not skip_position_embedding:
            self.pos_embedding = self.pos_embedding.to(x.device)
            x_pos_emb = self.pos_embedding(torch.arange(x.size()[1], device=x.device))
            x_emb += x_pos_emb
        return x_emb
