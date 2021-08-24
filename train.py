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
model = FrugalRnn(config['input_size'],config['n_memory'],torch.nn.Tanh(),
hidden_dims= config['hidden_dims'],
budget=config['budget']).to(device)
dataset = ParityDataset(config['n_samples_parity'],config['input_size'])

dataloader = DataLoader(dataset,batch_size=config['batch_size'])
optimizer = torch.optim.Adam(params = model.parameters(),lr=float(config['lr']))
loss_func = torch.nn.BCEWithLogitsLoss()


for epoch in range(0,config['n_epoch']):
    accuracy_tracker = IncrementalAverage()
    loss_tracker = IncrementalAverage()
    average_n_iter_tracker = IncrementalAverage()
    for index,batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        x,y = [a.to(device) for a in batch]
        probs,n_iters = model(x)
        batch_accuracy = (probs.round() == y.float()).float().mean().item()
        average_n_iter = n_iters.mean().item()
        loss = frugal_loss(probs,n_iters,y)
        loss.backward()
        optimizer.step(loss)
        accuracy_tracker.update(batch_accuracy)
        loss_tracker.update(loss.item())
        average_n_iter_tracker.update(average_n_iter)
        wandb.log({'batch_accuracy':batch_accuracy,'batch_loss':loss.item(),'batch_average_n_iter':average_n_iter})
    wandb.log({'accuracy':accuracy_tracker.value,'loss':loss_tracker.value,'average_n_iter':average_n_iter_tracker.value})