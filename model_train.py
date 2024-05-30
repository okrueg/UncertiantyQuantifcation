import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from model_architecture import basic_nn
from data_set_generation import generate_data, plot_data_static

def train(x,y):
    num_epochs = 40
    batch_size = 10
    hidden_dim = 40
    lr = 0.1

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = basic_nn(2,hidden_dim,1)
    loss_fn = torch.nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    #scheduler = ExponentialLR(optimizer,0.95)

    #-----Train ----
    model.train()
    for epoch in range(num_epochs):
        overall_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            x = model(x)
            x = x.view(-1)
            #print(abs(x-y))
            
            #print(x,y)
            loss = loss_fn(x, y)
            
            overall_loss += loss.item()
            
            loss.backward()

            optimizer.step()

        #scheduler.step()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

    #----Test ----
def test(x, model):
    x = torch.from_numpy(x).float()
    model.eval()
    with torch.no_grad():
        x = model(x)
    return x

# x,y = generate_data()
# plot_data_static(x,y)
# model = train(x,y)
# r = test(x_np, model)
# print(r)