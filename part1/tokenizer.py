import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)



def tokenize_text(df):
    df["text_tokens"] = df["text"].apply(word_tokenize)
    return df