import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import json

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def tokenize_text(df):
    df["text_tokens"] = df["text"].apply(word_tokenize)
    return df


class WordTokenizer:
    """Word-level tokenizer: nltk word_tokenize plus a vocabulary built from training texts"""

    def __init__(self, unk_token="[UNK]", lowercase=True):
        self.unk_token = unk_token
        self.lowercase = lowercase
        self.vocab = {unk_token: 0}

    def _split(self, text):
        return word_tokenize(text.lower() if self.lowercase else text)

    def train(self, corpus, vocab_size=None, min_frequency=1):
        counts = Counter(tok for text in corpus for tok in self._split(text))
        frequent = [t for t, c in counts.most_common() if c >= min_frequency]
        if vocab_size is not None:
            frequent = frequent[: vocab_size - 1]  # -1 leaves room for UNK
        for tok in frequent:
            self.vocab[tok] = len(self.vocab)
        return self

    def get_vocab(self):
        return self.vocab

    def encode(self, text):
        """Return the list of token ids for text"""
        unk = self.vocab[self.unk_token]
        return [self.vocab.get(t, unk) for t in self._split(text)]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    def load(self, path):
        with open(path) as f:
            self.vocab = json.load(f)
        return self
