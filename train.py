import torch
from models import FrugalRnn,frugal_loss
from dataset import ParityDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from utils import IncrementalAverage

config_path = 'configs/config.yaml'
wandb.init(project='FrugalAdaptiveComputation', entity='samme013',config= config_path)
config = wandb.config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FrugalRnn(n_input=  config['n_elems'],nonlin = torch.nn.Tanh(),n_hidden=config['n_hidden'],
budget=config['budget']).to(device)

optimizer = torch.optim.Adam(params = model.parameters(),lr=config['lr'])



for epoch in range(0,int(config['n_epoch'])):
    dataset = ParityDataset(config['batches_per_epoch']*config['batch_size'],n_elems=config['n_elems'])
    dataloader = DataLoader(dataset,batch_size=config['batch_size'])
    accuracy_tracker = IncrementalAverage()
    loss_tracker = IncrementalAverage()
    average_n_iter_tracker = IncrementalAverage()
    for index,batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        x,y = [a.to(device) for a in batch]
        probs,n_iters = model(x)
        batch_accuracy = (probs.sigmoid().round() == y.float()).float().mean().item()
        average_n_iter = n_iters.mean().item()
        loss = frugal_loss(probs,n_iters,y,budget=config['budget'],balance=config['balance']).mean()
        loss.backward()
        optimizer.step()
        accuracy_tracker.update(batch_accuracy)
        loss_tracker.update(loss.item())
        average_n_iter_tracker.update(average_n_iter)
    wandb.log({'accuracy':accuracy_tracker.value,'loss':loss_tracker.value,'average_n_iter':average_n_iter_tracker.value})