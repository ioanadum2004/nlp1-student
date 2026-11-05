from sklearn.neural_network import MLPClassifier


def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(50,),   
        activation='relu',
        solver='adam',
        alpha=1e-2,                  # L2 penalty 
        batch_size=512,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=5,          
        verbose=True,
        random_state=42,
    )