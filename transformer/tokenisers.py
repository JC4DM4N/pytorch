from typing import List
import torch
import re


class SimpleTokeniser:
    def __init__(self, max_len: int, vocab: List):
        self.max_len = max_len
        self.vocab = vocab

    def __call__(self, x: str):
        tokens = self.get_tokens(x)
        # replace tokens not in the vocabulary with [UNK]
        tokens = [token if token in self.vocab else "[UNK]" for token in tokens]
        # truncate to max_len
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        # add start and end characters
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # pad resultant token list
        while len(tokens) < self.max_len:
            tokens.append("[PAD]")
        # convert to integer values
        itokens = self.tokens_to_idx(tokens)
        return torch.tensor(itokens)

    def get_vocab(self, xs: List):
        special_chars = ["[CLS]", "[SEP]", "[UNK]"]
        all_tokens = []
        for x in xs:
            all_tokens += self.get_tokens(x)
        return list(set(all_tokens)) + special_chars

    @staticmethod
    def get_tokens(x: str):
        # remove special characters
        x = re.sub(r"[^A-Za-z0-9 ]+", '', x)
        # word tokenise and replace unknown words with [UNK]
        tokens = x.split()
        tokens = [token.lower() for token in tokens]
        return tokens

    def tokens_to_idx(self, tokens: List):
        return [self.vocab.index(token) for token in tokens]

    def idx_to_tokens(self, itokens: List):
        return [self.vocab[itoken] for itoken in itokens]
