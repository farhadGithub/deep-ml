import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    weights = weights.copy()
    m, n = X.shape
    if method == 'batch':
        batch_size = m
    elif method == 'stochastic':
        batch_size = 1
    for epoch in range(n_epochs):
        i = 0
        while i < m:
            X_batch = X[i:min(i+batch_size, m), :]
            y_batch = y[i:min(i+batch_size,m)]
            y_preds = X_batch @ weights
            weights -= (learning_rate*2/len(y_preds))*(y_preds - y_batch) @ X_batch
            i += batch_size

    return weights