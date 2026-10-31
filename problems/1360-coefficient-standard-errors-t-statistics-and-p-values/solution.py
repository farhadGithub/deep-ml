import math

import numpy as np


def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares fit with an intercept, plus inference on each coefficient.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (coef, se, t, p), each a length-(p+1) array, intercept first.
    """
    # Your code here
    n, p = X.shape
    D = np.column_stack([np.ones(len(X)), X])
    coefs = np.linalg.lstsq(D, y, rcond=None)[0]
    y_hat = D @ coefs
    sigma_2 = np.sum((y-y_hat)**2)/(n-p-1)
    var_coefs = sigma_2*np.linalg.inv(D.T @ D)
    se = np.sqrt(np.diag(var_coefs))
    t = coefs/se
    p = 2*(1-0.5*(1+np.array([math.erf(np.abs(ti)/math.sqrt(2)) for ti in t])))
    return (coefs, se, t, p)


