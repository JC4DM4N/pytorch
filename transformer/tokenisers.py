from typing import List, Dict
import torch
import re
import numpy as np


class SimpleTokeniser:
    def __init__(self, max_len: int, xs: List[str]):
        self.max_len = max_len
        self.vocab = self.get_vocab(xs)
        self.tokens_to_idx = self._tokens_to_idx()
        self.idx_to_tokens = self._idx_to_tokens()

    def __call__(self, x: str) -> torch.Tensor:
        tokens = self.get_tokens(x)
        # truncate to max_len
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        # add start and end characters
        tokens = np.concatenate((["[CLS]"], tokens, ["[SEP]"]))
        # pad resultant token list
        padding = ["[PAD]"]*(self.max_len - len(tokens))
        tokens = np.concatenate((tokens, padding))
        # convert to integer values
        itokens = self.convert_tokens_to_idx(tokens)
        return torch.tensor(itokens)

    def get_vocab(self, xs: List[str]) -> np.array:
        special_chars = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]
        all_tokens = []
        for x in xs:
            all_tokens += list(self.get_tokens(x))
        return np.array(sorted(list(set(all_tokens))) + special_chars)

    @staticmethod
    def get_tokens(x: str) -> np.array:
        # remove special characters
        x = re.sub(r"[^A-Za-z0-9 ]+", ' ', x)
        # word tokenise and replace unknown words with [UNK]
        tokens = x.lower().split()
        return np.array(tokens)

    def _tokens_to_idx(self) -> Dict[str, int]:
        return {token: i for i, token in enumerate(self.vocab)}

    def _idx_to_tokens(self) -> Dict[str, int]:
        return {i: token for i, token in enumerate(self.vocab)}

    def convert_tokens_to_idx(self, tokens: List[str]) -> List[int]:
        return [
            self.tokens_to_idx[token]
            if token in self.tokens_to_idx else self.tokens_to_idx["[UNK]"]
            for token in tokens
        ]

    def convert_idx_to_tokens(self, itokens: List[int]) -> List[str]:
        return [self.idx_to_tokens[itoken] for itoken in itokens]
