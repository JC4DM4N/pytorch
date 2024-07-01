import torch
import torch.nn as nn
import torch.functional as F


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


class PositionalEncoding:
    def __init__(self):
        pass

    def forward(self):
        pass


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
