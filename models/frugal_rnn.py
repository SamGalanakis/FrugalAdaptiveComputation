import torch
import torch.nn as nn
from models import MLP




class FrugalRnn(nn.Module):
    def __init__(self,n_hidden,n_memory,nonlin,budget):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_memory = n_memory
        self.nonlin = nonlin
        self.budget = budget


        self.iterator = MLP(n_hidden+n_memory,[64,64,64],n_hidden,nonlin)
        self.mem_updater = MLP(n_hidden+n_memory,[64,64,64],n_memory,nonlin)



    def forward(self,x):
        device =x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        halted_mask = torch.zeros((batch_size,),device=device,dtype=torch.long)
        memory = torch.zeros((batch_size,self.n_memory),dtype=dtype,device=device)
        memory[:,0] = memory[:,0]+ self.budget
        n_iters = torch.zeros((batch_size,),dtype=dtype,device=device)
        for index in range(self.budget):
            if halted.all():
                break

            x = 
            

