import random
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.functional import rotate

from model_architectures import BasicNN, DimensionIncreaser, BasicCNN
from data_set_generation import generate_data, plot_data_static
from torch.utils.data.sampler import  SubsetRandomSampler  #for validation test


def train_2d(x: np.ndarray,
          y: np.ndarray,
          shift: np.ndarray,
          model: BasicNN,
          dim_inc: DimensionIncreaser,
          num_epochs = 100,
          batch_size = 20,
          hidden_dim = 10,
          in_dims = 10,
          lr = 0.1):

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    shift = torch.from_numpy(shift).float()

    x_class1 = x[y == 1]

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.001)
    #scheduler = ExponentialLR(optimizer,0.95)

    drop_history = np.empty(hidden_dim)


    #-----Train ----
    model.train()
    for epoch in range(num_epochs):
        
        overall_loss = 0
        total_embeddings = None
        torch.save(model.state_dict(), f"models/model_{epoch}.path")

        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            #x = dim_inc(x)
            output = model(x)

            output = output.view(-1)

            loss = loss_fn(output, y)
            
            overall_loss += loss.item()
            
            loss.backward()

            optimizer.step()

        # get embeddings for dropout
        shift_dist = model.forward_embeddings(shift)
        total_embeddings = model.forward_embeddings(x_class1)

        # update dropout distributions
        if hasattr(model, 'dropout'):
            model.dropout.update_distribution(total_embeddings, train_dist=True)
            model.dropout.update_distribution(shift_dist, train_dist=False)

            #print(torch.mean(model.dropout.recent_mean_diff))

            drop_history = np.vstack((drop_history, model.dropout.recent_mean_diff))

            # decay alpha value to slowly introduce dropout
            model.dropout.alpha = model.dropout.alpha * .9

    return drop_history
        #scheduler.step()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


        
#----Test ----
def test_2d(x: np.ndarray, model: BasicNN, dim_inc: DimensionIncreaser):
    x = torch.from_numpy(x).float()
    model.eval()
    with torch.no_grad():
        #x = dim_inc(x)
        x = model(x)
    return x
