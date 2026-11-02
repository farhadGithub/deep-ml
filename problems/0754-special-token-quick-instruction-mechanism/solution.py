import numpy as np

def special_token_attention(Q, K, V, num_base, num_special):
	"""
	Q, K, V: array-like of shape (num_base + num_special, d)
	num_base: int, number of base tokens (L)
	num_special: int, number of appended special tokens (M)
	Returns: list of M lists (size d), attention outputs for the special tokens,
	rounded to 4 decimal places.
	"""
	ans = []
	Q = np.array(Q, dtype=float)
	K = np.array(K, dtype=float)
	V = np.array(V, dtype=float)
	dim = Q.shape[1]
	for i in range(num_special):
		Q_special = Q[num_base+i]
		K_special = np.vstack((K[:num_base], K[num_base+i]))
		V_special = np.vstack((V[:num_base], V[num_base+i]))
		scores = (Q_special @ K_special.T)/np.sqrt(dim)
		weights  = np.exp(scores)/np.sum(np.exp(scores))
		out = np.round(weights @ V_special, 4).tolist()
		ans.append(out)
	return ans
