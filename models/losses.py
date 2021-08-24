import torch



def frugal_loss(probs,n_iter,target,budget):
    return torch.abs(target-probs)#* (n_iter/budget)
