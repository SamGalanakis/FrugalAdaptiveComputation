import torch
from models import FrugalRnn,frugal_loss
from dataset import ParityDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


config_path = 'configs/config.yaml'
wandb.init(project='FrugalAdaptiveComputation', entity='samme013',config= config_path)
config = wandb.config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FrugalRnn(config['input_size'],config['n_memory'],torch.nn.Tanh(),
hidden_dims= config['hidden_dims'],
budget=config['budget']).to(device)
dataset = ParityDataset(config['n_samples_parity'],config['input_size'])

dataloader = DataLoader(dataset,batch_size=config['batch_size'])
optimizer = torch.optim.Adam(params = model.parameters(),lr=config['lr'])
loss_func = torch.nn.BCEWithLogitsLoss()
for index,batch in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    x,y = [a.to(device) for a in batch]
    probs,n_iters = model(x)
    accuracy = probs.round()
    loss = loss_frugal(probs,n_iters,y)
    loss.backward()
    optimizer.step(loss)