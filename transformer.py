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
    def __init__(self, embed_dim: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        # loop over the maximum number of tokens in the sequence
        for token in range(max_len):
            # loop over the length of each embedding
            for i in range(0, embed_dim, 2):
                pe[token, i] = math.sin(token / (10000 ** ((2 * i)/embed_dim)))
                pe[token, i + 1] = math.cos(token / (10000 ** ((2 * (i + 1))/embed_dim)))

        # include the batch size
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return self.pe


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int = 8, embed_dim: int = 512, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()

        self.d_k = embed_dim // heads               # single head dimension
        self.heads = heads

        self.query = nn.Linear(embed_dim, embed_dim)  # query vector - bias addition included
        self.key = nn.Linear(embed_dim, embed_dim)    # key vector - bias addition included
        self.value = nn.Linear(embed_dim, embed_dim)  # value vector - bias addition included

        self.out = nn.Linear(self.d_k*self.heads, self.d_k)

    def forward(self, X):
        """
        :param X: embedding matrix of shape (batch_size, max_len, embed_dim)
        :return:
        """
        batch_size = X.shape[0]
        max_len = X.shape[1]
        embed_dim = X.shape[2]

        # project X into query, key and value vectors
        query = self.query(X)       # (batch_size, max_len, embed_dim)
        key = self.key(X)           # (batch_size, max_len, embed_dim)
        value = self.value(X)       # (batch_size, max_len, embed_dim)

        # reshape vectors to shape (batch_size, max_len, heads, d_k)
        query = query.view(batch_size, max_len, self.heads, self.d_k)
        key = key.view(batch_size, max_len, self.heads, self.d_k)
        value = value.view(batch_size, max_len, self.heads, self.d_k)

        # then to (batch_size, heads, max_len, d_k)
        query = query.permute(batch_size, self.heads, max_len, self.d_k)
        key = key.permute(batch_size, self.heads, max_len, self.d_k)
        value = value.permute(batch_size, self.heads, max_len, self.d_k)

        # calculate self-attention
        attention_scores = F.softmax(
            torch.matmul(query, key.permute(0, 1, 3, 2))/math.sqrt(self.d_k)
        )                                                   # (batch_size, heads, max_len, max_len)
        context = torch.matmul(attention_scores, value)     # (batch_size, heads, max_len, d_k)

        # reshape back to (batch_size, max_len, embed_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, max_len, embed_dim)

        # final linear layer
        output = self.out(context)

        return output




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
