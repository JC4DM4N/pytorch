import os
import torch
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
import numpy as np
from pprint import pprint
from typing import Generator
import matplotlib.pyplot as plt

from transformer import Encoder
from tokenisers import SimpleTokeniser


class MaskedLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, num_layers: int, heads: int, dropout: float):
        super(MaskedLanguageModel, self).__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=max_len,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        # classifier layers
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.encoder(x)
        # Select only the embeddings of the masked positions
        masked_positions = mask.unsqueeze(-1).expand_as(x)  # shape (batch_size, seq_len, embed_dim)
        x_masked = x[masked_positions].view(-1, x.size(-1))  # shape (num_masked_tokens, embed_dim)

        logits = self.fc(x_masked)
        # probs = F.softmax(logits, dim=-1)
        return logits


def main():
    # CONFIG - training
    MODEL_NAME = "mlm_model"                # prefix for model files
    MODEL_OUTPUT_PATH = "mlm_model"         # path of directory to save model files
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
    train_size = 10000                      # sample size of training data to use
    test_size = 1000                        # sample size of test data to use

    # CONFIG - parameters for transformer model
    embed_dim = 256
    max_len = 128
    num_layers = 1
    heads = 1
    threshold = 0.5                         # probability threshold for assigning labels

    print("loading data...")
    # load dataset from datasets and take a sample - TinyStories data:
    # https://www.kaggle.com/datasets/thedevastator/tinystories-narrative-classification?resource=download
    train_df = pd.read_csv("archive_small/train.csv").dropna().sample(train_size, random_state=1)
    test_df = pd.read_csv("archive_small/train.csv").dropna().sample(test_size, random_state=1)

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

    def prepare_masked_texts(all_tokens: torch.Tensor, repeats: int = 1):
        masked_tokens_x, masked_tokens_y = [], []
        unwanted_token_ids = [
            tokeniser.tokens_to_idx[token] for token in ["[CLS]", "[SEP]", "[UNK]", "[PAD]"]
        ]
        mask_token_id = tokeniser.tokens_to_idx["[MASK]"]
        for _ in range(repeats):
            for tokens_ in all_tokens:
                masked_tokens = tokens_.detach().clone()
                replaced_token = unwanted_token_ids[0]
                while replaced_token in unwanted_token_ids:
                    imask = np.random.randint(max_len)
                    replaced_token = tokens_[imask].item()
                masked_tokens[imask] = mask_token_id
                masked_tokens_x.append(masked_tokens)
                masked_tokens_y.append(tokens_[imask])
        return torch.stack(masked_tokens_x), torch.stack(masked_tokens_y)

    train_tokens = train_df["text"].apply(tokeniser)
    train_tokens = torch.stack(train_tokens.to_list())

    test_tokens = test_df["text"].apply(tokeniser)
    test_tokens = torch.stack(test_tokens.to_list())

    def batch_data(x: torch.Tensor, batch_size: int):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(x), batch_size):
            yield x[i:i + batch_size]

    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.mkdir(MODEL_OUTPUT_PATH)

    # initialise models
    print("-"*50)
    print("initialising classifier...")
    model = MaskedLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_len=max_len,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    )
    # load saved model
    if load_epoch:
        print("loading model weights...")
        model.load_state_dict(torch.load(f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{load_epoch}.pth"))
    model = model.to(device)

    # define evaluation metrics
    loss_func = nn.CrossEntropyLoss()  # cross-entropy loss

    def custom_eval():
        pass

    import time
    if train_model:
        plt.ion()
        fig, ax = plt.subplots(2, 1, figsize=(5, 5))
        eval_metrics = {
            "train_loss": [],
            "eval_loss": [],
        }
        lines = []
        ilines = {}
        for i, ax_ in enumerate(ax.flatten()):
            metric = list(eval_metrics)[i]
            ax_.set_title(metric)
            ax_.set_ylim(0, 10)
            ax_.set_xlabel("epoch")
            ax_.set_xlim(load_epoch+1, load_epoch+num_epochs+1)
            lines += ax_.plot([], [], 'b-')
            ilines[metric] = i
        fig.canvas.draw()

        print("-"*50)
        print("training classifier...")
        tstart = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(load_epoch+1, load_epoch+num_epochs+1):
            for input_tokens in batch_data(train_tokens.detach().clone(), batch_size):
                x, y = prepare_masked_texts(input_tokens)
                x = x.to(device)
                y = y.to(device)
                mask = x == tokeniser.tokens_to_idx["[MASK]"]
                # forward pass
                preds = model(x, mask)
                loss = loss_func(preds, y)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

            if save_model_every and epoch % save_model_every == 0:
                torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}_epoch_{epoch}.pth")

            if eval_during_training:
                eval_metrics["train_loss"].append(loss.item())
                eval_preds = torch.tensor([])
                eval_labels = torch.tensor([])
                for input_tokens in batch_data(test_tokens.detach().clone(), batch_size):
                    x, y = prepare_masked_texts(input_tokens)
                    x = x.to(device)
                    y = y.to(device)
                    mask = x == tokeniser.tokens_to_idx["[MASK]"]
                    # forward pass
                    eval_preds = torch.cat((eval_preds, model(x, mask)))
                    eval_labels = torch.cat((eval_labels, y))
                eval_loss = loss_func(eval_preds, eval_labels.to(int))
                eval_metrics["eval_loss"].append(eval_loss.item())
                for metric in eval_metrics:
                    lines[ilines[metric]].set_data(range(load_epoch+1, epoch+1), eval_metrics[metric])
                fig.canvas.draw()
                plt.tight_layout()
                plt.pause(0.001)

        print("training complete...")
        print(f"training time: {time.time() - tstart}")
        # save output model
        torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/{MODEL_NAME}.pth")
        # save training logs
        plt.savefig(f"{MODEL_OUTPUT_PATH}/training_logs.png")

    if eval_model:
        print("-"*50)
        print("evaluating classifier... \n")
        for split, x in (["train", train_tokens], ["test", test_tokens]):
            eval_metrics = {}
            preds = torch.tensor([])
            labels = torch.tensor([])
            for input_tokens in batch_data(x.detach().clone(), batch_size):
                x, y = prepare_masked_texts(input_tokens)
                x = x.to(device)
                y = y.to(device)
                mask = x == tokeniser.tokens_to_idx["[MASK]"]
                # forward pass
                preds = torch.cat((preds, model(x, mask)))
                labels = torch.cat((labels, y))
            loss = loss_func(preds, labels.to(int))
            eval_metrics[f"cross_entropy_loss"] = loss.item()
            print(f"Evaluation metrics on {split} set:")
            pprint(eval_metrics)
            print("\n")


if __name__ == "__main__":
    main()
