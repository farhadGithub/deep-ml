import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    n, d  = X.shape
    min_g = float("Inf")
    best_feature_index = None
    best_threshold = None
    g_y = 1 - (np.sum(y==0)/n)**2 - (np.sum(y==1)/n)**2 
    for i in range(d):
        thresholds = np.sort(np.unique(X[:,i]))
        for threshold in thresholds: 
            n_left = np.sum(X[:,i]<=threshold)
            n_right = np.sum(X[:,i]>threshold)
            if n_left==0 or n_right==0:
                weighted_g = g_y
            else:
                g_left = 1 - (np.sum((y==0) & (X[:,i]<=threshold))/n_left)**2 -  (np.sum((y==1) & (X[:,i]<=threshold))/n_left)**2 
                g_right = 1 - (np.sum((y==0) & (X[:,i]>threshold))/n_right)**2 -  (np.sum((y==1) & (X[:,i]>threshold))/n_right)**2
                weighted_g = (n_left/n)*g_left + (n_right/n)*g_right
            if weighted_g<min_g:
                min_g = weighted_g
                best_feature_index = i
                best_threshold = threshold

    return best_feature_index, best_threshold