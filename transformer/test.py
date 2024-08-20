from datasets import load_dataset

from transformer import Transformer, Embedding, Encoder
from tokenisers import SimpleTokeniser

# load dataset from datasets
data = load_dataset("stanfordnlp/imdb")

# select sample of texts
texts = [item["text"] for item in data["train"].to_list()[0: 10]]
vocab = SimpleTokeniser(0, []).get_vocab(xs=texts)

# configuration for transformer model
src_vocab_size = len(vocab)
tgt_vocab_size = len(vocab)
embed_dim = 512
max_len = 128
num_layers = 2
heads = 2
output_dim = 128

print("-"*50)
print("Model Configuration:")
print(f"src_vocab_size: {src_vocab_size}")
print(f"tgt_vocab_size: {tgt_vocab_size}")
print(f"embed_dim: {embed_dim}")
print(f"max_len: {max_len}")
print(f"num_layers: {num_layers}")
print(f"heads: {heads}")
print(f"output_dim: {output_dim}")

# tokenise example text
tokeniser = SimpleTokeniser(max_len, vocab)
tokens = tokeniser(texts[0])

# test generate embeddings
print("-"*50)
print("testing Embedding class...")
embedding = Embedding(src_vocab_size, embed_dim)
embedded_tokens = embedding(tokens)
print(f"Embedding shape: {embedded_tokens.shape}")

print("-"*50)
print("testing Encoder class...")
# test encoder
encoder = Encoder(
    vocab_size=src_vocab_size,
    embed_dim=embed_dim,
    max_len=max_len,
    num_layers=num_layers,
    heads=heads,
)
encoded_text = encoder(tokens)
print(f"Encoding shape: {encoded_text.shape}")

# model = Transformer(
#     src_vocab_size=src_vocab_size,
#     tgt_vocab_size=tgt_vocab_size,
#     embed_dim=embed_dim,
#     max_len=max_len,
#     num_layers=num_layers,
#     heads=heads,
#     output_dim=output_dim
# )

# import torch
# from torch import nn
#
#
# criterion = nn.CrossEntropyLoss(ignore_index=0)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
# model.train()
#
# for epoch in range(100):
#     optimizer.zero_grad()
#     output = model(src_data, tgt_data[:, :-1])
#     loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

