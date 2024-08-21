from typing import List
import torch
import re
import numpy as np


class SimpleTokeniser:
    def __init__(self, max_len: int, xs: List[str]):
        self.max_len = max_len
        self.vocab = self.get_vocab(xs)

    def __call__(self, x: str) -> torch.Tensor:
        tokens = self.get_tokens(x)
        # replace tokens not in the vocabulary with [UNK]
        tokens[~np.in1d(tokens, self.vocab)] = "[UNK]"
        # truncate to max_len
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        # add start and end characters
        tokens = np.concatenate((["[CLS]"], tokens, ["[SEP]"]))
        # pad resultant token list
        padding = ["[PAD]"]*(self.max_len - len(tokens))
        tokens = np.concatenate((tokens, padding))
        # convert to integer values
        itokens = self.tokens_to_idx(tokens)
        return torch.tensor(itokens)

    def get_vocab(self, xs: List[str]) -> np.array:
        special_chars = ["[CLS]", "[SEP]", "[UNK]", "[PAD]"]
        all_tokens = []
        for x in xs:
            all_tokens += list(self.get_tokens(x))
        return np.array(list(set(all_tokens)) + special_chars)

    @staticmethod
    def get_tokens(x: str) -> np.array:
        # remove special characters
        x = re.sub(r"[^A-Za-z0-9 ]+", '', x)
        # word tokenise and replace unknown words with [UNK]
        tokens = x.lower().split()
        return np.array(tokens)

    def tokens_to_idx(self, tokens: List[str]) -> List[int]:
        return [np.where(self.vocab == token)[0][0] for token in tokens]

    def idx_to_tokens(self, itokens: List[int]) -> List[str]:
        return [self.vocab[itoken] for itoken in itokens]
