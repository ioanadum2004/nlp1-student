import numpy as np
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
            analyzer=lambda x: x,   # input is already a list of tokens
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

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector

class BagOfWords(Encoder):
    """Like TokenIndicator, but counts how often each token occurs instead of 0/1"""
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a vector of token counts"""
        vector = np.zeros(len(self))
        for token_id in self.tokenizer.encode(x).ids:
            vector[token_id] += 1
        return vector

class IdentityEncoder(Encoder):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[1]

    def __call__(self, i):
        return self.X[int(i)].toarray()[0]