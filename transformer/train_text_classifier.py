import os
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


def main():
    # CONFIG - training
    MODEL_NAME = "bert_classifier"          # prefix for model files
    MODEL_OUTPUT_PATH = "bert_classifier"   # path of directory to save model files
    num_epochs = 100                        # training epochs
    # batch_size = 4                          # batch size
    learning_rate = 1e-3                    # learning rate
    save_model_every = 10                   # no. epochs to save model weights file
    load_epoch = 0                          # file index to load
    device = torch.device("cpu")
    train_model = True                      # boolean to perform training
    eval_model = True                       # boolean to perform evaluation

    # CONFIG - parameters for transformer model
    embed_dim = 256
    max_len = 128
    num_layers = 2
    heads = 2
    output_dim = 1
    threshold = 0.5                         # probability threshold for assigning labels

    # load dataset from datasets and take a sample
    data = load_dataset("stanfordnlp/imdb")

    train_df = pd.DataFrame(data["train"]).sample(1000, random_state=1)
    test_df = pd.DataFrame(data["test"]).sample(1000, random_state=1)

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

    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.mkdir(MODEL_OUTPUT_PATH)

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

    # load saved model
    if load_epoch:
        print("loading model weights...")
        model.load_state_dict(torch.load(f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{load_epoch}.pth"))
    model = model.to(device)

    if train_model:
        print("-"*50)
        print("training classifier...")
        loss_func = nn.BCELoss()  # binary cross-entropy loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            preds = model(train_x)
            loss = loss_func(preds, train_y)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

            if save_model_every and epoch % save_model_every == 0:
                torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{epoch}.pth")

        print("training complete...")
        # save output model
        torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}.pth")

    if eval_model:
        print("-"*50)
        print("evaluating classifier... \n")

        for split, texts, labels in (["train", train_x, train_y], ["test", test_x, test_y]):

            probs = model(texts)
            preds = (probs > threshold)*1

            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            tn = ((preds == 0) & (labels == 0)).sum().item()

            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 0.5*(recall + precision)

            print(f"Evalutation metrics on {split} set:")
            print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            print(f"Recall: {recall}, Precision: {precision}, F1-Score: {f1} \n")


if __name__ == "__main__":
    main()
