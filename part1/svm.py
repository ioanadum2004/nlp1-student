import pickle
import numpy as np
from scipy.special import softmax
from sklearn.svm import SVC
from loglin import stack_vectors


def fit(
    training_set,
    encoder: "Encoder",
    save_to=None,
    kernel="rbf",
    C=1.0,
    gamma="scale",
    degree=3,
    cache_size=2000,
    max_iter=-1,
):
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [label for text, label in training_set]
    X_train_sparse = stack_vectors(training_vectors)

    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        cache_size=cache_size,
        max_iter=max_iter,
        decision_function_shape="ovr",
    )
    model.fit(X_train_sparse, training_labels)
    print(
        f"SVM ({kernel}) trained with {model.n_support_.sum()} support vectors "
        f"out of {X_train_sparse.shape[0]} training examples."
    )

    if save_to is not None:
        with open(f"{save_to}.pickle", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return model


def load(path):
    with open(f"{path}.pickle", "rb") as f:
        return pickle.load(f)


def predict(texts, model, encoder: "Encoder"):
    X = stack_vectors([encoder(text) for text in texts])
    return model.predict(X)


def predict_probs(texts, model, encoder: "Encoder"):
    X = stack_vectors([encoder(text) for text in texts])
    scores = model.decision_function(X)
    probs = np.zeros((X.shape[0], len(model.classes_)))
    probs[:, model.classes_] = softmax(scores, axis=1)  # columns indexed by label id
    return probs
