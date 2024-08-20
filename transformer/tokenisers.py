from typing import List
import torch


class SimpleTokeniser:
    def __init__(self, max_len: int, vocab: List):
        self.max_len = max_len
        self.vocab = vocab

    def __call__(self, x):
        tokens = [token if token in self.vocab else "[UNK]" for token in x.split()]
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        while len(tokens) < self.max_len:
            tokens.append("[PAD]")
        return torch.Tensor(tokens)
