from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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