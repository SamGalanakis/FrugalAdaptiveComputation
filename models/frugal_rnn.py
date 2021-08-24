import torch
import torch.nn as nn
from .nets import MLP




class FrugalRnn(nn.Module):
    def __init__(self,n_input,n_hidden,nonlin,hidden_dims,budget):
        super().__init__()
        self.n_hidden = n_hidden
  
        self.nonlin = nonlin
        self.budget = budget
        self.n_input = n_input


        self.iterator = MLP(n_hidden+n_input,hidden_dims,n_hidden+2,nonlin)
   


    def forward(self,x):
        device =x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        halted_mask = torch.zeros((batch_size,),device=device,dtype=torch.long).bool()
        hidden = torch.zeros((batch_size,self.n_hidden),dtype=dtype,device=device)
        hidden[:,0] = hidden[:,0]+ self.budget
        n_iters = torch.zeros((batch_size,),dtype=dtype,device=device)
        final_probs = torch.zeros((batch_size,),dtype=dtype,device=device)
        for index in range(self.budget):
            if halted_mask.all():
                break
            # Increment those not halted
            n_iters[~halted_mask]+= 1
            
            iterator_in = torch.cat((x[~halted_mask,...],hidden[~halted_mask,...]),dim=-1)
            iterator_out = self.iterator(iterator_in)
            
            probs,halt_val,hidden_out = iterator_out.split([1,1,self.n_hidden],dim=-1)
            hidden[~halted_mask,...] = hidden_out
            

            halt_outs = torch.sigmoid(halt_val).round().bool().squeeze()
            
            
            
            if (index == (self.budget-1)):
                # Set probs for all if at last iter
                final_probs[~halted_mask] = probs.squeeze()
            else:
                final_probs[~halted_mask] = final_probs[~halted_mask].where(~halt_outs,probs.squeeze() )
            halted_mask[~halted_mask] = halt_outs
        final_probs = final_probs.sigmoid()
        return final_probs,n_iters

