import numpy as np

def gradient_monte_carlo(episodes: list, alpha: float, gamma: float, n_epochs: int) -> list:
    """
    Learn value function weights using Gradient Monte Carlo with linear approximation.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state_features, reward) tuples.
                  state_features is a list of floats, reward is a float.
        alpha: Learning rate (step size)
        gamma: Discount factor
        n_epochs: Number of passes through all episodes
    
    Returns:
        List of learned weight values, each rounded to 4 decimal places.
    """
    num_features = len(episodes[0][0][0])
    weights = np.zeros((num_features))
    for epoch in range(n_epochs):
       for episode in episodes:
            Gt = 0
            for features, reward in episode[::-1]:
                features = np.array(features)
                Gt = gamma*Gt + reward
                v_hat = weights @ features
                weights = weights + alpha*(Gt - v_hat)*features
    return weights
