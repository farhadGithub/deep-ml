import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	def sigmoid(x: np.ndarray) -> np.ndarray:
		return 1./(1 + np.exp(-x))

	n = len(labels)
	updated_weights = initial_weights.copy()
	updated_bias = initial_bias
	mse_values = []
	for epoch in range(epochs):
		z = features@updated_weights + updated_bias
		preds = sigmoid(z)
		mse = np.sum((preds-labels)**2)/n
		mse_values.append(round(mse,4))
		updated_bias -= (2./n)*learning_rate*np.sum(preds*(preds-labels)*(1-preds))
		updated_weights -= (2./n)*learning_rate*(preds*(preds-labels)*(1-preds)).T@features
		
	return np.round(updated_weights,4), np.round(updated_bias,4), mse_values
