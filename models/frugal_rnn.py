import torch
import torch.nn as nn
from .nets import MLP




class FrugalRnn(nn.Module):
    def __init__(self,n_hidden,n_memory,nonlin,hidden_dims,budget):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_memory = n_memory
        self.nonlin = nonlin
        self.budget = budget


        self.iterator = MLP(n_hidden+n_memory,hidden_dims,n_hidden+n_memory+2,nonlin)
   


    def forward(self,x):
        device =x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        halted_mask = torch.zeros((batch_size,),device=device,dtype=torch.long).bool()
        memory = torch.zeros((batch_size,self.n_memory),dtype=dtype,device=device)
        memory[:,0] = memory[:,0]+ self.budget
        n_iters = torch.zeros((batch_size,),dtype=dtype,device=device)
        final_probs = torch.zeros((batch_size,),dtype=dtype,device=device)
        for index in range(self.budget):
            if halted_mask.all():
                break
            
            
            iterator_in = torch.cat((x[~halted_mask,...],memory[~halted_mask,...]),dim=-1)
            iterator_out = self.iterator(iterator_in)
            
            probs,halt_val,hidden_out,memory_out = iterator_out.split([1,1,self.n_hidden,self.n_memory],dim=-1)
            x[~halted_mask,...] = hidden_out
            memory[~halted_mask,...] = memory_out

            halt_outs = torch.sigmoid(halt_val).round().bool().squeeze()
            final_probs[~halted_mask] = final_probs[~halted_mask].where(~halt_outs,probs.squeeze() )
            halted_mask[~halted_mask] = halt_outs

        final_probs = final_probs.sigmoid()
        return final_probs,n_iters

