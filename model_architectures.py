import torch
import numpy as np
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler


class basic_nn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(basic_nn, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.dropout = shifted_dist_dropout()
        
        self.ReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input(x)

        x = self.ReLU(x)

        x = self.hidden1(x)

        x = self.ReLU(x)

        x = self.dropout.forward(x)

        x = self.output(x)

        x = self.sigmoid(x)

        return x
    
    def forwardEmbeddings(self, x):
        with torch.no_grad():
            x = self.input(x)

            x = self.ReLU(x)

            x = self.hidden1(x)

            x = self.ReLU(x)
            
            return x


class shifted_dist_dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1
        self.training_distribution = None
        self.shifted_distribution = None
        self.scaler = MinMaxScaler()
    
    def update_distribution(self, channel_matrix, train_dist):
        var_mean = torch.var_mean(channel_matrix, dim=0) # tuple(var,mean)
        if train_dist:
            self.training_distribution = var_mean
        else:
            self.shifted_distribution = var_mean

    def get_mask(self, x):
        mean_dif = abs(self.training_distribution[1] - self.shifted_distribution[1])
        
        scaled_mean_dif = torch.div(mean_dif, self.training_distribution[0]) # divide by varience

        scaled_mean_dif = scaled_mean_dif.reshape(-1,1)

        scaled_mean_dif = self.scaler.fit_transform(scaled_mean_dif)

        scaled_mean_dif = torch.from_numpy(scaled_mean_dif).reshape(-1)

        mask = (torch.rand(*mean_dif.shape) + self.alpha ) > scaled_mean_dif

        return mask
        
    def forward(self, x):
        if self.training and self.training_distribution != None:
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