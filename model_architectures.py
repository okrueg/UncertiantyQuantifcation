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

        self.dropout = shifted_dist_dropout(hidden_dim)
        
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

class basic_cnn(nn.Module):
    def __init__(self, num_classes):
        super(basic_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)
    
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1)

        #self.dropout = nn.Dropout(0.5)
        self.dropout = Class_confusion_dropout(0.2)
        
        self.fc1 = nn.Linear(2048, 1024)


        self.fc2 = nn.Linear(1024, 20)
        self.fc3 = nn.Linear(20, num_classes)

        self.flatten = nn.Flatten()
        self.ReLU = nn.LeakyReLU(0.2)
        #self.softmax = nn.Softmax(dim=1)

        self._max_norm_val = 3
        self._eps = 1e-8

    def forward(self, x: torch.Tensor, y=None):
        assert x.isnan().any().item() == False
        
        x = self.ReLU(self.conv1(x))

        x = self.pool1(x)
        
        x = self.ReLU(self.conv2(x))

        x = self.pool2(x)
        
        x = self.flatten(x)

        #self.fc1.weight.data = self._max_norm(self.fc1.weight.data)
        x = self.ReLU(self.fc1(x))

        #x = self.dropout(x)

        #self.fc2.weight.data = self._max_norm(self.fc2.weight.data)
        x = self.ReLU(self.fc2(x))

        x = self.dropout(x)
        if y != None:
            self.dropout.get_stats(x,y)
        
        x = self.fc3(x)
        
        return x

    #https://github.com/kevinzakka/pytorch-goodies#max-norm-constraint
    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

class Class_confusion_dropout(nn.Module):
    def __init__(self, drop_percent):
        super().__init__()

        self.weight_matrix = None
        self.prev_output = None
        self.engaged = True

        self.drop_percent = drop_percent

        self.info = None

    def get_mask(self, x):
        #retrieve the indicies of the 2 highest model predictions
        top_ind = torch.topk(self.prev_output, k=2, dim=1)[1]

        #select the weights associated with those model predictions
        selected_weight_cols = self.weight_matrix[top_ind] #Shape: batch, 2, feature size

        selected_weight_diffs = abs(selected_weight_cols[:, 0, :] - selected_weight_cols[:, 1, :]) #Shape: batch, feature size

        #weight * channel, Higher Score means higherdrop rate
        scores = x * selected_weight_diffs #Shape: batch, feature

        #Number of channels to drop
        num_dropped = int(x.shape[1] * self.drop_percent)

        #select num_dropped num channels with the highest score 
        dropped_channels = torch.topk(scores, k=num_dropped, dim=1, largest=True)[1]

        mask = torch.ones_like(x).bool()

        for batch_indx, _ in enumerate(mask):
            mask[batch_indx][dropped_channels[batch_indx]] = False

        return mask
    
    def get_stats(self, x, y):
        #retrieve the indicies of the 2 highest model predictions
        top_ind = torch.topk(self.prev_output, k=2, dim=1)[1]

        #select the weights associated with those model predictions
        selected_weight_cols = self.weight_matrix[top_ind] #Shape: batch, 2, feature size

        selected_weight_diffs = abs(selected_weight_cols[:, 0, :] - selected_weight_cols[:, 1, :]).detach() #Shape: batch, feature size

        #weight * channel, Higher Score means higherdrop rate
        scores = x * selected_weight_diffs #Shape: batch, feature

        #Number of channels to drop
        num_dropped = int(x.shape[1] * self.drop_percent)

        #select num_dropped num channels with the highest score 
        dropped_channels = torch.topk(scores, k=num_dropped, dim=1, largest=True)[1]

        mask = torch.ones_like(x)

        for batch_indx, _ in enumerate(mask):
            mask[batch_indx][dropped_channels[batch_indx]] = 0

        # Make epoch compatable
        y = torch.unsqueeze(y, dim=0)
        dropped_channels = torch.unsqueeze(mask, dim=0)
        top_ind = torch.unsqueeze(top_ind, dim=0)
        selected_weight_diffs = torch.unsqueeze(selected_weight_diffs, dim=0)

        if self.info == None: # if this is the first batch
            self.info = {"labels": y,
                         "dropped_channels": dropped_channels, 
                         "selected_weight_indexs": top_ind, 
                         "weight_diffs": selected_weight_diffs}
        else:
            # Concat along the batch dimension
            self.info["labels"] = torch.cat((self.info["labels"], y), dim= 1) # Shape: [1,batch]
            self.info["dropped_channels"] = torch.cat((self.info["dropped_channels"], dropped_channels), dim= 1) # Shape: [1,batch, num_dropped]
            self.info["selected_weight_indexs"] = torch.cat((self.info["selected_weight_indexs"], top_ind), dim= 1)
            self.info["weight_diffs"] = torch.cat((self.info["weight_diffs"], selected_weight_diffs), dim= 1)

        # for x in self.info.values():
        #     print(x.shape)

        
    def forward(self, x: torch.Tensor):
        if self.training:
            if self.prev_output != None and self.engaged:
                mask = self.get_mask(x)
                x = x * mask
        else:
            x = x * (1 - self.drop_percent) # keep expected value
        return x


class shifted_dist_dropout(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.alpha = 1
        self.training_distribution = torch.rand(2,input_dim, dtype= torch.float64)
        self.shifted_distribution = torch.rand(2,input_dim, dtype = torch.float64)
        self.scaler = MinMaxScaler()

        self.recent_mean_diff = None
    
    def update_distribution(self, channel_matrix, train_dist):
        var_mean = torch.var_mean(channel_matrix, dim=0) # tuple(var,mean)
        if train_dist:
            self.training_distribution = var_mean
        else:
            self.shifted_distribution = var_mean

    def get_mask(self, x, verbose = True):
        mean_dif = abs(self.training_distribution[1] - self.shifted_distribution[1])
        
        scaled_mean_dif = torch.div(mean_dif, self.training_distribution[0]) # divide by varience

        #print("before ",scaled_mean_dif)

        scaled_mean_dif = scaled_mean_dif.reshape(-1,1)

        scaled_mean_dif = self.scaler.fit_transform(scaled_mean_dif)

        scaled_mean_dif = torch.from_numpy(scaled_mean_dif).reshape(-1).type(torch.float64)
        #print("after ",scaled_mean_dif)

        if verbose:
            self.recent_mean_diff = scaled_mean_dif

        #print(scaled_mean_dif)

        mask = (torch.rand(*mean_dif.shape) + self.alpha ) > scaled_mean_dif

        return mask
        
    def forward(self, x: torch.Tensor):
        if self.training:
            mask = self.get_mask(x)
            print(x.shape)
            x = x * mask
        else:
            #print(self.recent_mean_diff.float())
            x = x * (1 - self.recent_mean_diff.float())
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