import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    
    Args:
        log_probs_chosen_policy: Log-probs of chosen responses under policy
        log_probs_rejected_policy: Log-probs of rejected responses under policy
        log_probs_chosen_ref: Log-probs of chosen responses under reference model
        log_probs_rejected_ref: Log-probs of rejected responses under reference model
        beta: Temperature parameter for KL constraint strength
    
    Returns:
        Dictionary with 'loss', 'chosen_rewards', and 'rejected_rewards'
    """
    def sigmoid(x):
        return 1./(1+np.exp([-xi for xi in x]))
    chosen_rewards = [beta*(p-r) for p,r in zip(log_probs_chosen_policy,log_probs_chosen_ref)]
    rejected_rewards = [beta*(p-r) for p, r in zip(log_probs_rejected_policy, log_probs_rejected_ref)]
    diff = [c-r for c,r in zip(chosen_rewards, rejected_rewards)]
    loss = -np.mean(np.log(sigmoid(diff)))
    return {'loss': round(loss,4), 'chosen_rewards': [round(c,4) for c in chosen_rewards], 'rejected_rewards': [round(r,4) for r in rejected_rewards]}
