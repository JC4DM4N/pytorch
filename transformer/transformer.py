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

    def forward(self, query, key, value, mask=None):
        """
        :param query: Q embedding matrix of shape (batch_size, max_len, embed_dim)
        :param key: K embedding matrix of shape (batch_size, max_len, embed_dim)
        :param value: V embedding matrix of shape (batch_size, max_len, embed_dim)
        :param mask: triangular mask for use in the decoder
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

        # calculate self-attention (batch_size, heads, max_len, max_len)
        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2))/math.sqrt(self.d_k)

        # Masking for the decoder
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        # apply softmax
        attention_scores = F.softmax(attention_scores)

        # multiply by value to get context (batch_size, heads, max_len, d_k)
        context = torch.matmul(attention_scores, value)

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

    def forward(self, query, key, value, mask=None):
        # multi-head layer
        attention_out = self.multi_head(query, key, value, mask)
        dropout1_out = self.dropout1(attention_out)
        attention_out_residual = dropout1_out + query  # residual connection for first layer
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

    def forward(self, query, key, value, mask=None):
        attention_out = self.multi_head(query, key, value, mask)
        attention_norm = self.norm1(attention_out)
        query = self.dropout(attention_norm) + query  # residual connection
        out = self.transformer_block(query, key, value, mask)
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, num_layers: int, heads: int, output_dim: int):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(heads, embed_dim)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.softmax = F.softmax()

    def forward(self, x, enc_out, mask=None):
        x = self.embedding(x) + self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, mask)
        x = self.softmax(self.fc(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, num_layers: int, heads: int, output_dim: int):
        super(Transformer, self).__init__()

        self.encoder = Encoder(vocab_size, embed_dim, max_len, num_layers, heads)
        self.decoder = Decoder(vocab_size, embed_dim, max_len, num_layers, heads, output_dim)

    def make_tgt_mask(self, tgt: torch.Tensor):
        """
        Create a target mask for use in the decoder, which will be a sparse lower triangular matrix
            filled with ones.

        Args:
            tgt (torch.Tensor): Target sequence.

        Returns:
            tgt_mask (torch.Tensor): Target mask.
        """
        batch_size, tgt_len = tgt.shape
        # returns the lower triangular part of matrix filled with ones
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(
            batch_size, 1, tgt_len, tgt_len
        )
        return tgt_mask

    def decode(self, tgt: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(tgt, enc_out, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Forward pass of the Transformer model.

        Args:
            src (torch.Tensor): Input to the encoder.
            tgt (torch.Tensor): Input to the decoder.

        Returns:
            outputs (torch.Tensor): Final vector representing probabilities of each target word.
        """
        enc_output = self.encoder(src)
        tgt_mask = self.make_tgt_mask(tgt)
        out = self.decode(tgt, enc_output, tgt_mask)
        return out


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
