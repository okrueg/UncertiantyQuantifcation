from bayesian_torch.layers.variational_layers import LinearReparameterization, Conv2dReparameterization
import torch.nn.functional as F
from torch import nn
import torch
import sinData
import numpy as np
class BNN(nn.Module):
    def __init__(self, in_channels, in_feat, out_feat):
        super(BNN, self).__init__()

        self.conv1 = Conv2dReparameterization(in_channels, out_channels=32, kernel_size=3)

        self.conv2 = Conv2dReparameterization(32, 64, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = LinearReparameterization(12544, 6272)

        self.fc2 = LinearReparameterization(6272, 10)

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

        x = self.flatten(x)

        x, kl = self.fc1(x)
        kl_sum += kl
        x = self.activation(x)

        x, kl = self.fc2(x)
        kl_sum += kl

        #output = x

        output = F.log_softmax(x, dim=1)

        return output, kl_sum

class basic(nn.Module):
    def __init__(self):
        super(basic, self).__init__()
        self.input_layer = nn.Linear(in_features=1, out_features=10)

        self.output_layer = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x, y= None):
        kl_sum = 0
        x = self.input_layer(x)

        x = F.relu(x)
        x = self.output_layer(x)

        output = F.log_softmax(x, dim=1)
        #output = x
        return output, kl_sum



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

# test(model,10, sinData.test_loader)
