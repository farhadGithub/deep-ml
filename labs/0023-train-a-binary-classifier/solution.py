import numpy as np
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    learning_rate = 0.01
    steps = 100
    n_samples = len(X_train)
    n_val = len(X_val)
    n_features = X_train.shape[1]
    W = np.random.normal(loc=0, scale=1, size=n_features)

    def sigmoid(X):
        return 1/(1+np.exp(-np.dot(W, X.T)))

    def predict(X):
        h = sigmoid(X)
        return h>=0.5

    prev_loss = 0
    for step in range(steps):
        loss = 0
        for Xi, yi in zip(X_train, y_train):
            W -= -learning_rate*(yi-sigmoid(Xi))*Xi
            if yi==1:
                loss += -np.log(sigmoid(Xi))
            else:
                loss += -(1-np.log(sigmoid(Xi)))
        if (abs(loss-prev_loss)/loss)<0.01:
            #print(f'training done with loss {loss}.')
            break

        prev_loss = loss
        #print(f'step {step} with loss {loss}.')
    
    return predict