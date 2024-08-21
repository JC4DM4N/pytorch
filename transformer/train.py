import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd

from transformer import Encoder
from tokenisers import SimpleTokeniser


class BERTClassifier(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, num_layers: int, heads: int, num_classes: int):
        super(BERTClassifier, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=max_len,
            num_layers=num_layers,
            heads=heads
        )
        # classifier layers
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        # average pooling of embeddings over full sequence -
        # Note BERT uses [CLS] embedding as a representation of the full sentence
        # but we haven't accounted for that here.
        x = x.mean(dim=1)
        x = F.sigmoid(self.fc(x))
        return x


# load dataset from datasets and take a sample
data = load_dataset("stanfordnlp/imdb")

train_df = pd.DataFrame(data["train"]).sample(1000).reset_index(drop=True)
test_df = pd.DataFrame(data["test"]).sample(1000).reset_index(drop=True)

# configuration parameters for transformer model
embed_dim = 256
max_len = 128
num_layers = 2
heads = 2
output_dim = 1

# train tokeniser using vocabulary in the train set only
tokeniser = SimpleTokeniser(max_len, train_df["text"])

vocab = tokeniser.vocab
vocab_size = len(vocab)
print("-"*50)
print("Model Configuration:")
print(f"vocab_size: {vocab_size}")
print(f"embed_dim: {embed_dim}")
print(f"max_len: {max_len}")
print(f"num_layers: {num_layers}")
print(f"heads: {heads}")
print(f"output_dim: {output_dim}")

train_tokens = train_df["text"].apply(tokeniser)
train_x = torch.stack(train_tokens.to_list())
train_y = torch.tensor(train_df["label"].astype(float).to_list()).view(train_df.shape[0], 1)


test_tokens = test_df["text"].apply(tokeniser)
test_x = torch.stack(test_tokens.to_list())
test_y = torch.tensor(test_df["label"].astype(float).to_list()).view(test_df.shape[0], 1)

# # test encoder class
print("-"*50)
print("initialising classifier...")
model = BERTClassifier(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    max_len=max_len,
    num_layers=num_layers,
    heads=heads,
    num_classes=output_dim
)

print("-"*50)
print("training classifier...")
loss_func = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(100):
    optimizer.zero_grad()
    preds = model(train_x)
    loss = loss_func(preds, test_y)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
