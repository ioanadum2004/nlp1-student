import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def _as_text(x):
    if isinstance(x, (list, tuple)):
        return " ".join(map(str, x))
    return str(x)


def tokenize_text(df):
    df["text_tokens"] = df["text"].map(_as_text).apply(word_tokenize)
    return df