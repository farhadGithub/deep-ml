import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size, itos, seed, max_tokens):
    rng = np.random.default_rng(seed)
    vocab_size = len(itos)
    context = [0 for _ in range(block_size)]
    C = np.array(C, dtype=float)
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)
    ans = ''
    for _ in range(max_tokens):
        e  = C[context,:].flatten() # shape: context_window x emb_dim
        h = np.tanh(e @ W1 + b1)
        logits = h @ W2 + b2
        logits = np.exp(logits - np.max(logits))
        logits = logits / np.sum(logits)
        ix = rng.choice(vocab_size, p=logits)
        if ix==0:
            break
        ans += itos[ix]
        context = context[1:] + [ix]
    return ans
    

    