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
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        # loop over the maximum number of tokens in the sequence
        for token in range(max_len):
            # loop over the length of each embedding
            for i in range(0, embed_dim, 2):
                pe[token, i] = math.sin(token / (10000 ** ((2 * i)/embed_dim)))
                pe[token, i + 1] = math.cos(token / (10000 ** ((2 * (i + 1))/embed_dim)))

        # include the batch size
        self.pe = pe.unsqueeze(0)
        self.pe = torch.autograd.Variable(self.pe[:, :max_len], requires_grad=False)

    def forward(self, x):
        return self.pe


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int = 8, embed_dim: int = 512):
        super(MultiHeadedAttention, self).__init__()

        self.d_k = embed_dim // heads               # single head dimension
        self.heads = heads

        self.query = nn.Linear(embed_dim, embed_dim)  # query vector - bias addition included
        self.key = nn.Linear(embed_dim, embed_dim)    # key vector - bias addition included
        self.value = nn.Linear(embed_dim, embed_dim)  # value vector - bias addition included

        self.out = nn.Linear(self.d_k*self.heads, self.d_k)

    def forward(self, query, key, value):
        """
        :param x: embedding matrix of shape (batch_size, max_len, embed_dim)
        :return:
        """
        batch_size = query.shape[0]
        max_len = query.shape[1]
        embed_dim = query.shape[2]

        # project x into query, key and value vectors
        query = self.query(query)       # (batch_size, max_len, embed_dim)
        key = self.key(key)             # (batch_size, max_len, embed_dim)
        value = self.value(value)       # (batch_size, max_len, embed_dim)

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


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int):
        super(TransformerBlock, self).__init__()
        self.multi_head = MultiHeadedAttention(heads, embed_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim*4)
        self.fc2 = nn.Linear(embed_dim*4, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        # multi-head layer
        attention_out = self.multi_head(x, x, x)
        dropout1_out = self.dropout1(attention_out)
        attention_out_residual = dropout1_out + x  # residual connection for first layer
        norm1_out = self.norm1(attention_out_residual)
        # fc layers
        fc1_out = self.dropout(F.Relu(self.fc1(norm1_out)))
        fc2_out = self.fc2(fc1_out)
        fc_out_redidual = fc2_out + norm1_out  # residual connection for second layer
        out = self.norm2(fc_out_redidual)
        return out


class Encoder(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, num_layers: int, heads: int):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(num_layers)])

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, x, x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, heads: int, embed_dim: int):
        super(DecoderBlock, self).__init__()
        self.multi_head = MultiHeadedAttention(heads, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, heads)

    def forward(self, query, key, value):
        attention_out = self.multi_head(query, key, value)
        attention_norm = self.norm1(attention_out)
        query = self.dropout(attention_norm + query)
        out = self.transformer_block(query, key, value)
        return out


class Decoder:
    def __init__(self):
        pass

    def forward(self):
        pass


if __name__ == "__main__":

    def _print_model_layers():
        model = Encoder(
            vocab_size=100,
            embed_dim=512,
            max_len=128,
            num_layers=12,
            heads=3
        )
        print(model)

    # bert_model = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-uncased")
    _print_model_layers()
