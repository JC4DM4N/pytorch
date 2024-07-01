import torch
import torch.nn as nn
import torch.functional as F
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embedding vectors
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim).float()

        # loop over the maximum number of tokens in the sequence
        for pos in range(max_len):
            # loop over the length of each embedding
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embed_dim)))

        # include the batch size
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe


class SelfAttention:
    def __init__(self):
        pass

    def forward(self):
        pass


class Encoder:
    def __init__(self):
        pass

    def forward(self):
        pass


class Decoder:
    def __init__(self):
        pass

    def forward(self):
        pass
