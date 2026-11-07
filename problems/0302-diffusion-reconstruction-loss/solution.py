import numpy as np

def diffusion_loss(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray, predicted_noise: np.ndarray) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: True noise array (same shape as x_0)
        predicted_noise: Model's predicted noise (same shape as x_0)
    
    Returns:
        Mean squared error loss (float)
    """
    alpha_hat_t = 1
    for t in range(1, t+1):
        beta_t = beta_start + ((t-1)/(num_timesteps-1))*(beta_end-beta_start)
        alpha_t = 1 - beta_t
        alpha_hat_t *= alpha_t
        x_t = np.sqrt(alpha_hat_t)*x_0 + np.sqrt(1-alpha_hat_t)*noise
    x_hat_0 = (x_t -  np.sqrt(1-alpha_hat_t)*predicted_noise)/np.sqrt(alpha_hat_t) 
    return np.mean((x_0 -x_hat_0)**2)