import torch
import torch.nn as nn
from .nets import MLP




class FrugalRnn(nn.Module):
    def __init__(self,n_input,nonlin,n_hidden,budget):
        super().__init__()
      
        self.n_hidden = n_hidden
        self.nonlin = nonlin
        self.budget = budget
        self.n_input = n_input

  
        self.iterator = torch.nn.GRUCell(n_input,n_hidden)

        self.stopper = torch.nn.Linear(n_hidden,1)

        self.predictor = torch.nn.Linear(n_hidden,1)


   


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
                #print(f'All halted')
                break
            # Increment those not halted
            n_iters[~halted_mask]+= 1
            
            
            new_hidden = self.iterator(x[~halted_mask,...],hidden[~halted_mask,...])
            

            hidden[~halted_mask,...] = new_hidden
            
            stop_vals = self.stopper(new_hidden)
            halt_outs = torch.sigmoid(stop_vals).round().bool().squeeze()
            
            probs = self.predictor(new_hidden)
            
            if (index == (self.budget-1)):
                # Set probs for all if at last iter
                final_probs[~halted_mask] = probs.squeeze()
            else:
                final_probs[~halted_mask] = final_probs[~halted_mask].where(~halt_outs,probs.squeeze() )
            halted_mask[~halted_mask] = halt_outs
        
        return final_probs,n_iters

