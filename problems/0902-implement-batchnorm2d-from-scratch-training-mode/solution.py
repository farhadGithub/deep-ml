import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    mu = x.mean(keepdim=True, axis=[0,2,3])
    var = x.var(keepdim=True, axis=[0,2,3])
    x_hat = (x-mu)/torch.sqrt((var+eps))
    y = beta[None,:, None, None] + gamma[None,:, None, None]*x_hat
    return y