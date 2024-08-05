from bayesian_torch.layers.variational_layers import LinearReparameterization, Conv2dReparameterization
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


class BNN(nn.Module):
    def __init__(self, in_channels):
        super(BNN, self).__init__()

        self.conv1 = Conv2dReparameterization(in_channels, out_channels=32, kernel_size=3)

        self.conv2 = Conv2dReparameterization(32, 64, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = LinearReparameterization(12544, 10)

        self.fc2 = LinearReparameterization(10, 10)

        self.activation= torch.nn.LeakyReLU()

    def forward(self, x, y=None):
        kl_sum = 0
        x, kl = self.conv1(x)
        kl_sum += kl
        x = self.activation(x)

        x, kl = self.conv2(x)
        kl_sum += kl
        x = self.activation(x)

        x = self.pool(x)
        #print(x.shape)

        x = self.flatten(x)
        
        x, kl = self.fc1(x)
        kl_sum += kl
        x = self.activation(x)

        x, kl = self.fc2(x)
        kl_sum += kl

        return x, kl_sum

class DNN(nn.Module):
    def __init__(self, in_channels):
        super(DNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels=32, kernel_size=3)

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = torch.nn.Linear(12544, 2048)

        self.fc2 = torch.nn.Linear(2048, 10)

        self.activation= torch.nn.LeakyReLU()

        self.activations= None

    def forward(self, x, y=None):

        x = self.conv1(x)

        #print(x)

        x = self.activation(x)

        x = self.conv2(x)

        x = self.activation(x)

        x = self.pool(x)

        x = self.flatten(x)
        
        x = self.fc1(x)

        x = self.activation(x)

        x = self.fc2(x) 

        return x


# epochs = 50
# model = BNN()
# model.to('mps')

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.90)
# for epoch in range(1, epochs + 1):
#     train(model,
#           num_mc=10,
#           batch_size=32,
#           device='mps',
#           train_loader=sinData.train_loader,
#           optimizer=optimizer,
#           epoch = epoch)
#     scheduler.step()

