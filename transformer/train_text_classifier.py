import os
import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
from pprint import pprint
from typing import Generator
import matplotlib.pyplot as plt

from transformer import Encoder
from tokenisers import SimpleTokeniser


class BERTClassifier(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, num_layers: int, heads: int, num_classes: int, dropout: float):
        super(BERTClassifier, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=max_len,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        # classifier layers
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.encoder(x))
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
    num_epochs = 50                         # training epochs
    batch_size = 64                         # batch size
    learning_rate = 1e-4                    # learning rate
    dropout = 0.2                           # global dropout used across all modules
    save_model_every = 10                   # no. epochs to save model weights file
    load_epoch = 0                          # file index to load
    device = torch.device("cpu")
    train_model = True                      # boolean to perform training
    eval_model = True                       # boolean to perform evaluation
    eval_during_training = True             # whether to calculate and save eval metrics at each epoch
    train_size = 10000                       # sample size of training data to use
    test_size = 1000                        # sample size of test data to use

    # CONFIG - parameters for transformer model
    embed_dim = 256
    max_len = 128
    num_layers = 1
    heads = 1
    output_dim = 1
    threshold = 0.5                         # probability threshold for assigning labels

    # load dataset from datasets and take a sample
    data = load_dataset("stanfordnlp/imdb")

    train_df = pd.DataFrame(data["train"]).sample(train_size, random_state=1)
    test_df = pd.DataFrame(data["test"]).sample(test_size, random_state=1)

    # train tokeniser using vocabulary in the train set only
    tokeniser = SimpleTokeniser(max_len, train_df["text"])

    vocab = tokeniser.vocab
    vocab_size = len(vocab)
    print("-"*50)
    print("Model Configuration: \n")
    print(f"vocab_size: {vocab_size}")
    print(f"embed_dim: {embed_dim}")
    print(f"max_len: {max_len}")
    print(f"num_layers: {num_layers}")
    print(f"heads: {heads}")
    print(f"output_dim: {output_dim}")

    print("-"*50)
    print("Training Configuration: \n")
    print(f"model_name: {MODEL_NAME}")
    print(f"learning_rate: {learning_rate}")
    print(f"num_epochs: {num_epochs}")
    print(f"batch_size: {batch_size}")
    print(f"dropout: {dropout}")

    # tokenise texts
    print("-"*50)
    print("tokenising texts...")
    train_tokens = train_df["text"].apply(tokeniser)
    train_x = torch.stack(train_tokens.to_list())
    train_y = torch.tensor(train_df["label"].astype(float).to_list()).view(train_df.shape[0], 1)

    test_tokens = test_df["text"].apply(tokeniser)
    test_x = torch.stack(test_tokens.to_list())
    test_y = torch.tensor(test_df["label"].astype(float).to_list()).view(test_df.shape[0], 1)

    print(tokeniser.tokens_to_idx["and"])
    print(tokeniser.tokens_to_idx["movie"])
    print(tokeniser.tokens_to_idx["funny"])

    def batch_data(x: torch.Tensor, y: torch.Tensor, batch_size: int):
        """Yield successive n-sized chunks from lst."""
        assert x.shape[0] == y.shape[0]
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size], y[i:i + batch_size]

    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.mkdir(MODEL_OUTPUT_PATH)

    # initialise models
    print("-"*50)
    print("initialising classifier...")
    model = BERTClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_len=max_len,
        num_layers=num_layers,
        heads=heads,
        num_classes=output_dim,
        dropout=dropout
    )
    # load saved model
    if load_epoch:
        print("loading model weights...")
        model.load_state_dict(torch.load(f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{load_epoch}.pth"))
    model = model.to(device)

    # define evaluation metrics
    loss_func = nn.BCELoss()  # binary cross-entropy loss

    def custom_eval(model: nn.Module, data_loader: Generator, threshold: float, loss_func: torch.nn):
        probs = torch.Tensor([])
        labels = torch.Tensor([])
        preds = torch.Tensor([])

        for x, y in data_loader:
            probs = torch.cat((probs, model(x)))
            labels = torch.cat((labels, y))
        preds = torch.cat((preds, (probs > threshold) * 1))

        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        recall = tp / (tp + fn) if tp else 0
        precision = tp / (tp + fp) if tp else 0
        f1 = 0.5 * (recall + precision)
        BCE_loss = loss_func(probs, labels)

        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Recall": recall,
            "Precision": precision,
            "F1-Score": f1,
            "loss": BCE_loss.item()
        }

    if train_model:
        plt.ion()
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))
        eval_metrics = {
            "train_loss": [],
            "loss": [],
            "TP": [],
            "FP": [],
            "FN": [],
            "TN": [],
            "Recall": [],
            "Precision": [],
            "F1-Score": []
        }
        lines = []
        ilines = {}
        for i, ax_ in enumerate(ax.flatten()):
            metric = list(eval_metrics)[i]
            if metric == "train_loss":
                ax_.set_title(metric)
            else:
                ax_.set_title(f"eval_{metric}")
            if metric in ["TP", "FP", "FN", "TN"]:
                ax_.set_ylim(0, test_size)
            else:
                ax_.set_ylim(0, 1)
            ax_.set_xlabel("epoch")
            ax_.set_xlim(0, num_epochs)
            lines += ax_.plot([], [], 'b-')
            ilines[metric] = i
        fig.canvas.draw()

        print("-"*50)
        print("training classifier...")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(1, num_epochs+1):
            for x, y in batch_data(train_x, train_y, batch_size):
                x = x.to(device)
                y = y.to(device)
                # forward pass
                preds = model(x)
                loss = loss_func(preds, y)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

            if save_model_every and epoch % save_model_every == 0:
                torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{epoch}.pth")

            if eval_during_training:
                test_loader = batch_data(test_x, test_y, batch_size)
                eval_scores = custom_eval(model, test_loader, threshold, loss_func)
                eval_scores["train_loss"] = loss.item()
                for metric in eval_scores:
                    eval_metrics[metric].append(eval_scores[metric])
                    lines[ilines[metric]].set_data(range(1, epoch+1), eval_metrics[metric])
                fig.canvas.draw()
                plt.tight_layout()
                plt.pause(0.001)

        print("training complete...")
        # save output model
        torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}.pth")
        # save training logs
        plt.savefig(f"{MODEL_OUTPUT_PATH}/training_logs.png")


    if eval_model:
        print("-"*50)
        print("evaluating classifier... \n")
        for split, x, y in (["train", train_x, train_y], ["test", test_x, test_y]):
            batched_data = batch_data(x, y, batch_size)
            scores = custom_eval(model, batched_data, threshold, loss_func)
            print(f"Evaluation metrics on {split} set:")
            pprint(scores)
            print("\n")


if __name__ == "__main__":
    main()
