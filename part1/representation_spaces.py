from sklearn.feature_extraction.text import CountVectorizer

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
    #return df["token_indicators"]

def bag_of_words(text_tokens, vocab):
    vectorizer = CountVectorizer(
        vocabulary=vocab,
        analyzer=lambda x: x,
        binary=False, 
    )
    return list(vectorizer.transform(text_tokens))