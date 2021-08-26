import torch


bce_loss = torch.nn.BCEWithLogitsLoss()
def frugal_loss(probs,n_iter,target,budget):
    
    return bce_loss(probs,target.float())
