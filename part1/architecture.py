from sklearn.neural_network import MLPClassifier


def build_mlp():
    return MLPClassifier(
    hidden_layer_sizes=(100,),  # One hidden layer with 100 neurons
    activation='relu',          # Activation
    solver='adam',              # Optimizer
    max_iter=200,               # Maximum iterations
    random_state=42,             # Random state for reproducibility
    verbose=True,
    batch_size=1024,
)