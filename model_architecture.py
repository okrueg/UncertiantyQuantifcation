import torch
import numpy as np
import torch.nn as nn


class basic_nn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(basic_nn, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        # self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=0.25)
        
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input(x)

        x = self.ReLU(x)

        #x = self.dropout(x)

        x = self.hidden1(x)

        x = self.ReLU(x)

        # x = self.hidden2(x)

        # x = self.ReLU(x)

        x = self.output(x)

        x = self.sigmoid(x)

        return x