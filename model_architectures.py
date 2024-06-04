import torch
import numpy as np
import torch.nn as nn


class basic_nn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(basic_nn, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.dropout = shifted_dist_dropout()
        
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input(x)

        x = self.ReLU(x)

        #x = self.dropout(x)

        x = self.hidden1(x)

        x = self.ReLU(x)

        x = self.output(x)

        x = self.sigmoid(x)

        return x
    
    def forwardEmbeddings(self, x):
        with torch.no_grad():
            x = self.input(x)

            x = self.ReLU(x)

            x = self.hidden1(x)
            
            return x


class shifted_dist_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_distribution = None
        self.shifted_distribution = None
        
    def get_mask(self, x):
        mask = torch.rand(*x.shape)<=self.p
        return mask
    
    def update_distribution(self, channel_matrix, train_dist):
        var_mean = torch.var_mean(channel_matrix, dim=0) # tuple(var,mean)
        if train_dist:
            self.training_distribution = var_mean
        else:
            self.shifted_distribution = var_mean
        
    def forward(self, x):
        if self.training:
            mask = self.get_mask(x)
            x = x * mask
        return x
















class dimension_increaser(nn.Module):
    def __init__(self, input_dim, output_dim, use_activation=True):
        super(dimension_increaser, self).__init__()
        self.scramble_layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.activation = nn.Sigmoid()
        self.use_activation = use_activation
        self.requires_grad_(requires_grad=False)

    def forward(self, x):
        x = self.scramble_layer(x)
        if self.use_activation:
            x = self.activation(x)
        return x