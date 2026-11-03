import numpy as np

def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    """
    Build a decision tree for regression and predict on test data.
    
    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        X_test: Test features, shape (m_samples, n_features)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
    
    Returns:
        List of predictions for X_test, rounded to 4 decimal places
    """
    
    def train(X_partition, y_partition, depth, node_id):
        tree[node_id] = {}
        n_samples = y_partition.shape[0]
        mse = (1./n_samples)*np.sum((y_partition-y_partition.mean())**2)   
        if mse==0 or depth >= max_depth or n_samples < min_samples_split:
            tree[node_id]['value'] = y_partition.mean()
            return
        max_gain = 0
        best_threshold = None
        best_feature_id = None
        best_mask = None
        for feature_id in range(n_features):
            unique_values = np.sort(np.unique(X_partition[:,feature_id]))
            for j in range(1, len(unique_values)):
                threshold = (unique_values[j-1]+unique_values[j])/2
                mask = X_partition[:,feature_id]<=threshold
                n_l = y_partition[mask].shape[0]
                n_r = y_partition[~mask].shape[0]
                mse_l = (1./n_l)*np.sum((y_partition[mask] - y_partition[mask].mean())**2)
                mse_r = (1./n_r)*np.sum((y_partition[~mask] - y_partition[~mask].mean())**2)
                gain = mse - ((n_l*mse_l + n_r*mse_r)/(n_l+n_r))
                if gain>=max_gain:
                    max_gain = gain
                    best_feature_id = feature_id
                    best_threshold = threshold
                    best_mask = mask
        
        tree[node_id]['feature_id'] = best_feature_id 
        tree[node_id]['threshold'] = best_threshold

        left_node_id = node_id+'L'
        tree[node_id]['left_node_id'] = left_node_id
        train(X_partition[best_mask], y_partition[best_mask], depth+1, left_node_id)
        
        right_node_id = node_id+'R'
        tree[node_id]['right_node_id'] = right_node_id
        train(X_partition[~best_mask], y_partition[~best_mask], depth+1, right_node_id)

        return

    def predict(tree, X):
        y_pred = []
        for x in X:
            node_id = 'root'
            while True:
                if tree[node_id].get('value') is not None:
                    y_pred.append(round(tree[node_id]['value'], 4))
                    break
                feature_id = tree[node_id]['feature_id']
                threshold = tree[node_id]['threshold']
                if x[feature_id] <= threshold:
                    node_id = tree[node_id]['left_node_id']
                else:
                    node_id = tree[node_id]['right_node_id']

        return y_pred

    tree = {}  #node_id:[feature_id, threshold, left_node_id, right_node_id, value]
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    _, n_features = X_train.shape
    train(X_train, y_train, 0, 'root') 
    return predict(tree, X_test)    
