import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    # Your code here
    ans  = idx.copy()
    for i in range(max_new_tokens):
        cropped_idx = np.array([ans[-context_size:]])
        logits = np.array(model(cropped_idx))
        logits = logits[0,-1,:]
        new_id = int(np.argmax(logits))
        idx.append(new_id)
        ans.append(new_id)
    return ans
