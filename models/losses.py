import torch


bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
def frugal_loss(probs,n_iter,target,budget,balance):
    target_loss = bce_loss(probs,target.float())
    penalization = (n_iter-1)/budget
    loss = balance * target_loss + (1-balance)* penalization
    return loss
