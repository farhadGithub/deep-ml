import numpy as np


def pairwise_sq_dists(X, C):
    # TODO: ||x||^2 + ||c||^2 - 2 x.c via one matrix product, clipped at 0
    X_norms = np.sum(X ** 2, axis=1)[:, None]
    C_norms = np.sum(C ** 2, axis=1)
    return np.maximum(X_norms + C_norms - 2*X @ C.T, 0)


def assign_clusters(X, C):
    # TODO: argmin over centroids
    distances = pairwise_sq_dists(X, C)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, C):
    # TODO: mean of each cluster's points; empty clusters keep their centroid
    sums = np.zeros_like(C)
    np.add.at(sums, labels, X)
    counts = np.bincount(labels, minlength=len(C))
    C_new = C.copy()
    mask = counts > 0
    C_new[mask] = sums[mask]/counts[mask, None]
    return C_new


def kmeans(X, C0, max_iter=100, tol=1e-6):
    # TODO: iterate until the largest centroid movement < tol; return (C, labels, n_iter)
    C = np.array(C0, dtype=float, copy=True)
    for n_iter in range(1, max_iter+1):
        labels = assign_clusters(X, C)
        C_new = update_centroids(X, labels, C)
        movement = np.linalg.norm(C_new - C, axis=1)
        C = C_new
        if np.max(movement) < tol:
            break
    labels = assign_clusters(X, C)
    return (C, labels, n_iter)

def inertia(X, C, labels):
    ans = 0
    for label in sorted(np.unique(labels)):
        Ci = C[label]
        ssd = np.sum(pairwise_sq_dists(X[labels==label], Ci[None,:]))
        ans += ssd
    return float(ans)
