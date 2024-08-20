from datasets import load_dataset

from transformer import Transformer, Embedding, Encoder, Decoder
from tokenisers import SimpleTokeniser

# load dataset from datasets
data = load_dataset("stanfordnlp/imdb")

# select sample of texts
texts = [item["text"] for item in data["train"].to_list()[0: 10]]
vocab = SimpleTokeniser(0, []).get_vocab(xs=texts)

# configuration parameters for transformer model
src_vocab_size = len(vocab)
tgt_vocab_size = len(vocab)
embed_dim = 512
max_len = 128
num_layers = 2
heads = 2
output_dim = len(vocab)
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
tokens = tokeniser(texts[0]).view(1, max_len)

# test generate embeddings
print("-"*50)
print("testing Embedding class...")
embedding = Embedding(src_vocab_size, embed_dim)
embedded_tokens = embedding(tokens)
print(f"Embedding shape: {embedded_tokens.shape}")
print(embedded_tokens)

# test encoder class
print("-"*50)
print("testing Encoder class...")
encoder = Encoder(
    vocab_size=src_vocab_size,
    embed_dim=embed_dim,
    max_len=max_len,
    num_layers=num_layers,
    heads=heads,
)
encoded_text = encoder(tokens)
print(f"Encoding shape: {encoded_text.shape}")
print(encoded_text)

# test decoder class
print("-"*50)
print("testing Decoder class...")
decoder = Decoder(
    vocab_size=tgt_vocab_size,
    embed_dim=embed_dim,
    max_len=max_len,
    num_layers=num_layers,
    heads=heads,
    output_dim=output_dim
)
decoded_output = decoder(tokens, encoded_text)
print(f"Decoded output shape: {decoded_output.shape}")
print(decoded_output)

# test full transformer implementation
print("-"*50)
print("testing full Transformer class...")
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    embed_dim=embed_dim,
    max_len=max_len,
    num_layers=num_layers,
    heads=heads,
    output_dim=output_dim
)
output_scores = model(tokens, tokens)
print(f"Output shape: {output_scores.shape}")
print(output_scores)

# generate some next-token predictions with the randomly initialised implementation
print("-"*50)
print("Predictions with randomly initialised weights... \n")
scores = output_scores[0]
tokens = tokens[0]
# randomly select some tokens to use for prediction
for itoken in [10, 20, 50]:
    print(
        f"Context tokens: {tokeniser.idx_to_tokens(tokens[:itoken])}"
    )
    next_predicted_token_idx = int(scores[itoken].argmax())
    print(
        f"Next predicted token: {tokeniser.idx_to_tokens([next_predicted_token_idx])} \n"
    )
