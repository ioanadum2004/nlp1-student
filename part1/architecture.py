from sklearn.neural_network import MLPClassifier


def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        batch_size=512,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=3,
        verbose=True,
        random_state=42,
    )