import torch



def frugal_loss(probs,n_iter,target):
    probs_loss = target-probs
