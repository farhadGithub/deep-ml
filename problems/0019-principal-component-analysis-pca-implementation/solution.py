import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    data = data - np.mean(data, axis=0)
    data = data/np.std(data, axis=0)
    covariance_matrix = data.T @ data
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvectors = eigenvectors[:, -k:][:, ::-1]    
    for i in range(eigenvectors.shape[1]):
        first_nonzero = np.nonzero(eigenvectors[:, i])[0][0]
        if eigenvectors[first_nonzero, i] < 0:
            eigenvectors[:, i] *= -1
    eigenvectors = np.round(eigenvectors, 4)
    return eigenvectors
    
