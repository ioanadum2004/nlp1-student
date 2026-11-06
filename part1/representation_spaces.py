import numpy as np
from scipy.sparse import csr_matrix
from clslib import Encoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tokenizers import Tokenizer


def vocab(df):
    vocab = {x for l in df["text_tokens"] for x in l}
    return vocab


def token_indicators(text_tokens, vocab):
    vectorizer = CountVectorizer(
        vocabulary=vocab,
        analyzer=lambda x: x,
        binary=True,
        lowercase=False,
    )
    return list(vectorizer.transform(text_tokens))
    # return vectorizer.transform(text_tokens) # this return a matrix instead of the list w the rows
    # return df["token_indicators"]


def bag_of_words(text_tokens, vocab):
    vectorizer = CountVectorizer(
        vocabulary=vocab,
        analyzer=lambda x: x,
        binary=False,
        lowercase=False,
    )
    return list(vectorizer.transform(text_tokens))
    # return vectorizer.transform(text_tokens)


def tfidf(text_tokens, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer=lambda x: x,  # input is already a list of tokens
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            norm="l2",
        )
        X = vectorizer.fit_transform(text_tokens)
    # for val and test input with vectorizer from training
    else:
        X = vectorizer.transform(text_tokens)
    return X, vectorizer


class TokenIndicator(Encoder):
    "As provided in the tutorial"

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> csr_matrix:
        """Represent x as a sparse 1 x D row with 1s at the tokens appearing in x"""
        cols = sorted(set(self.tokenizer.encode(x)))
        rows = [0] * len(cols)
        data = [1.0] * len(cols)
        return csr_matrix((data, (rows, cols)), shape=(1, len(self)))


class BagOfWords(Encoder):
    """Like TokenIndicator, but counts how often each token occurs instead of 0/1"""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> csr_matrix:
        """Represent x as a sparse 1 x D row of token counts"""
        cols = self.tokenizer.encode(x)
        rows = [0] * len(cols)
        data = [1.0] * len(cols)
        return csr_matrix((data, (rows, cols)), shape=(1, len(self)))


class Tfidf(Encoder):
    """Like TokenIndicator, but uses tf-idf instead of 0/1"""

    def __init__(self, tokenizer: Tokenizer, norm: bool = True):
        self.tokenizer = tokenizer
        self.norm = norm #L2
        self.idf = None           # set by fit()

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def fit(self, corpus) -> "Tfidf":
        """Learn idf weights from the training texts (an iterable of str)."""
        df = np.zeros(len(self))      # number of documents containing each token
        n_docs = 0
        for text in corpus:
            ids = set(self.tokenizer.encode(text).ids)  
            df[list(ids)] += 1
            n_docs += 1
        # smoothed idf, same formula as sklearn: ln((1 + n) / (1 + df)) + 1
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1
        return self

    def __call__(self, x: str) -> "np.array":
        """Represent x as a vector of tf-idf values"""
        if self.idf is None:
            raise RuntimeError("Call fit(train_texts) before encoding.")

        ids = self.tokenizer.encode(x).ids
        tf = np.bincount(ids, minlength=len(self)).astype(float)   # raw term counts
        vector = tf * self.idf                                      # tf-idf

        if self.norm:
            length = np.linalg.norm(vector)
            if length > 0:
                vector /= length
        return vector

class IdentityEncoder(Encoder):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[1]

    def __call__(self, i):
        return self.X[int(i)]
