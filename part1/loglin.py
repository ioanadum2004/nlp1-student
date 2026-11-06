import scipy.sparse
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm.auto import tqdm


def stack_vectors(vectors):
    """Stack encoder outputs into one sparse N x D matrix.
    Works for sparse 1 x D rows (representation_spaces encoders) and for
    dense 1D numpy vectors (the tutorial's encoders)."""
    if scipy.sparse.issparse(vectors[0]):
        return scipy.sparse.vstack(vectors, format="csr")
    return scipy.sparse.csr_matrix(vectors)

def load(path):
    """
    Load a logistic model from {path}.pickle and return it.
    """
    with open(f"{path}.pickle", 'rb') as f:
        logistic_model = pickle.load(f)
    return logistic_model

def fit(training_set, encoder: 'Encoder', save_to=None, max_iter=100, n_jobs=-1, tol=1e-4, solver="saga"):
    """
    Constructs, trains and returns a softmax regressor.
    
    training_set: list of text-label pairs (a text is a string, a label is a 0-based integer)
    encoder: an instance of the Encoder class (such as TokenIndicator)    
    save_to: (str) if given, save the model to {save_to}.pickle
    """
    # Convert the training and validation sets to vectors
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [label for text, label in training_set]

    # Stack the encoder outputs into one sparse N x D matrix
    X_train_sparse = stack_vectors(training_vectors)

    # Train the logistic model
    logistic_model = LogisticRegression(solver=solver, max_iter=max_iter, tol=tol, n_jobs=n_jobs)
    logistic_model.fit(X_train_sparse, training_labels)

    n_iter = int(max(logistic_model.n_iter_))
    if n_iter >= max_iter:
        print(f"WARNING: logistic regression did NOT converge ({n_iter}/{max_iter} iterations). Increase max_iter.")
    else:
        print(f"Logistic regression converged in {n_iter} iterations.")

    if save_to is not None:        
        with open(f"{save_to}.pickle", 'wb') as f:
            pickle.dump(logistic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return logistic_model        

def predict(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    validation_vectors = [encoder(text) for text in texts]
    X_val_sparse = stack_vectors(validation_vectors)
    predictions = model.predict(X_val_sparse)
    return predictions

def predict_probs(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    validation_vectors = [encoder(text) for text in texts]
    X_val_sparse = stack_vectors(validation_vectors)
    probs = model.predict_proba(X_val_sparse)
    return probs