import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.array(X, dtype=float)
    n_samples, n_features = X.shape
    y = np.array(y)
    current_predictions = np.array(current_predictions)
    residuals = y - current_predictions
    min_mse = np.inf
    best_threshold = 0
    best_feature = 0
    best_left_value = np.mean(residuals)
    best_right_value = np.mean(residuals)
    for i in range(n_features):
        values = np.sort(np.unique(X[:,i]))
            
        for j in range(1, len(values)):
            threshold = (values[j-1]+values[j])/2
            mask = X[:,i]<=threshold
            left_value = np.mean(residuals[mask]) 
            left_residuals = residuals[mask] - left_value
            right_value = np.mean(residuals[~mask]) 
            right_residuals = residuals[~mask] - right_value
            mse = np.sum(left_residuals**2) + np.sum(right_residuals**2)
            if mse < min_mse:
                min_mse = mse
                best_feature = i
                best_threshold = threshold
                best_left_value = left_value
                best_right_value = right_value 

    # update current_predictions
    updated_predictions = []
    for i in range(n_samples):
        if X[i][best_feature] <= best_threshold:
            updated_prediction = current_predictions[i] + learning_rate * best_left_value
        else:
            updated_prediction = current_predictions[i] + learning_rate * best_right_value
        updated_predictions.append(float(updated_prediction))

    return updated_predictions