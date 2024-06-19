'''
'''
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

class BasicNN(nn.Module):
    '''
    ultra simple NN
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicNN, self).__init__()

        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        self.dropout = shiftedDistDropout(hidden_dim)

        self.ReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        '''
        forwards
        '''
        x = self.input(x)

        x = self.ReLU(x)

        x = self.hidden1(x)

        x = self.ReLU(x)

        x = self.dropout.forward(x)

        x = self.output(x)

        x = self.sigmoid(x)

        return x


    def forward_embeddings(self, x):
        '''
        forwards early
        '''
        with torch.no_grad():
            x = self.input(x)

            x = self.ReLU(x)

            x = self.hidden1(x)

            x = self.ReLU(x)

            return x


class BasicCNN(nn.Module):
    '''
    a simple CNN archetechture
    '''
    def __init__(self, num_classes: int, feature_size: int, dropout_prob: float):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1)

        #self.dropout = nn.Dropout(dropout_prob)
        self.dropout = ConfusionDropout(dropout_prob, DropoutDataHandler())

        self.fc1 = nn.Linear(2048, 1024)


        self.fc2 = nn.Linear(1024, feature_size)
        self.fc3 = nn.Linear(feature_size, num_classes)

        self.flatten = nn.Flatten()
        self.ReLU = nn.LeakyReLU(0.2)
        #self.softmax = nn.Softmax(dim=1)

        self._max_norm_val = 3
        self._eps = 1e-8

    def forward(self, x: torch.Tensor, y=None):
        '''
        forwards through model
        '''
        assert x.isnan().any().item() is False

        x = self.ReLU(self.conv1(x))

        x = self.pool1(x)

        x = self.ReLU(self.conv2(x))

        x = self.pool2(x)

        x = self.flatten(x)

        self.fc1.weight.data = self._max_norm(self.fc1.weight.data)
        x = self.ReLU(self.fc1(x))

        #x = self.dropout(x)

        self.fc2.weight.data = self._max_norm(self.fc2.weight.data)
        x = self.ReLU(self.fc2(x))

        x = self.dropout(x, y)
        #x = self.dropout(x)

        self.fc3.weight.data = self._max_norm(self.fc3.weight.data)
        x = self.fc3(x)

        return x

    #https://github.com/kevinzakka/pytorch-goodies#max-norm-constraint
    def _max_norm(self, w):
        norm = w.norm(3, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

class ConfusionDropout(nn.Module):
    '''
    A special form of dropout to challenge the model by causing class confusion
    '''
    def __init__(self, drop_percent, drop_handler = None):
        super().__init__()

        self.weight_matrix = torch.empty(0)
        self.prev_output = None
        self.engaged = True

        self.drop_percent = drop_percent
        self.drop_handeler = drop_handler

    def get_mask(self, x, y = None):
        '''
        retrieves mask for doc string
        '''
        #retrieve the indicies of the 2 highest model predictions
        top_ind = torch.topk(self.prev_output, k=2, dim=1)[1]

        #select the weights associated with those model predictions
        selected_weight_cols = self.weight_matrix[top_ind] #Shape: batch, 2, feature size

        selected_weight_diffs = selected_weight_cols[:, 0, :] - selected_weight_cols[:, 1, :] #Shape: batch, feature size

        #weight * channel, Higher Score means higherdrop rate
        scores = abs(x * selected_weight_diffs) #Shape: batch, feature

        #Number of channels to drop
        num_dropped = int(x.shape[1] * self.drop_percent)

        #select num_dropped num channels with the highest score 
        dropped_channels = torch.topk(scores, k=num_dropped, dim=1, largest=True)[1]

        mask = torch.ones_like(x).bool()

        for batch_indx, _ in enumerate(mask):
            mask[batch_indx][dropped_channels[batch_indx]] = False

        if y is not None:
            self.drop_handeler.store_forwardpass(y.detach().to('cpu'),
                                                 mask.detach().to('cpu'),
                                                 top_ind.detach().to('cpu'),
                                                 selected_weight_diffs.detach().to('cpu'))

        return mask


    def forward(self, x: torch.Tensor, y = None):
        '''
        forwards dropout
        '''
        if self.training:
            if self.prev_output is not None:
                mask = self.get_mask(x, y)
                x = x * mask
        else:
            x = x * (1 - self.drop_percent) # keep expected value
        return x


class DropoutDataHandler():
    '''
    for collecting info in our special dropout
    '''
    def __init__(self):
        self.batch_info = None
        self.epoch_info = None


    def store_forwardpass(self, y, mask, top_ind, selected_weight_diffs):
        '''
        Store info about dropout's pass
        '''

        # Make epoch compatable
        y = torch.unsqueeze(y, dim=0)
        dropped_channels = torch.unsqueeze(mask, dim=0)
        top_ind = torch.unsqueeze(top_ind, dim=0)
        selected_weight_diffs = torch.unsqueeze(selected_weight_diffs, dim=0)
        if self.batch_info is None: # if this is the first batch
            self.batch_info = {"labels": y,
                         "dropped_channels": dropped_channels, 
                         "selected_weight_indexs": top_ind, 
                         "weight_diffs": selected_weight_diffs}
        else:
            # Concat along the batch dimension
            self.batch_info["labels"] = torch.cat((self.batch_info["labels"], y), dim= 1) # Shape: [1,batch]
            self.batch_info["dropped_channels"] = torch.cat((self.batch_info["dropped_channels"], dropped_channels), dim= 1) # Shape: [1,batch, num_dropped]
            self.batch_info["selected_weight_indexs"] = torch.cat((self.batch_info["selected_weight_indexs"], top_ind), dim= 1)
            self.batch_info["weight_diffs"] = torch.cat((self.batch_info["weight_diffs"], selected_weight_diffs), dim= 1)

        # for x in self.batch_info.values():
        #     print(x.shape)


    def add_epoch(self):
        '''
        stack data from this epoch
        '''
        if self.epoch_info is None:
            self.epoch_info = self.batch_info.copy()
        else:
            for item in self.batch_info:
                self.epoch_info[item] = torch.vstack((self.epoch_info[item], self.batch_info[item]))            

        self.batch_info = None


    def seperate_batch_info(self, drop_info_epochs):
        '''
        Depricated
        '''
        seperated = [None for x in range(10)]
        for label in range (10):
            label_indices = torch.nonzero((drop_info_epochs["labels"] == label),as_tuple=True)[1] #PROBLEM CHILD
            print(drop_info_epochs["labels"].shape, label_indices.shape)

            dropped_channels_per_label = drop_info_epochs["dropped_channels"][:, label_indices, :]
            chosen_labels_per_label =  drop_info_epochs["selected_weight_indexs"][:, label_indices, :]
            weight_diffs_per_label =  drop_info_epochs["weight_diffs"][:, label_indices, :]

            print(dropped_channels_per_label.shape)

            seperated[label] = {"dropped_channels": dropped_channels_per_label,
                            "selected_weight_indexs": chosen_labels_per_label,
                            "weight_diffs": weight_diffs_per_label}

        return seperated


    def forward_info(self, epoch: int, label: int, inquiry: str):
        '''
        retrieves the data of a specific epoch
        '''
        print(self.epoch_info["dropped_channels"].shape)
        label_indices = torch.nonzero((self.epoch_info["labels"][epoch] == label), as_tuple=True)[0]

        print(label_indices.shape)

        selected_per_label = self.epoch_info[inquiry][epoch][label_indices, :]

        print(selected_per_label.shape)


        return selected_per_label


class shiftedDistDropout(nn.Module):
    '''
    A failed dropout :(
    '''
    def __init__(self, input_dim):
        super().__init__()
        self.alpha = 1
        self.training_distribution = torch.rand(2,input_dim, dtype= torch.float64)
        self.shifted_distribution = torch.rand(2,input_dim, dtype = torch.float64)
        self.scaler = MinMaxScaler()

        self.recent_mean_diff = None


    def update_distribution(self, channel_matrix, train_dist):
        '''
        Updates distributions of the dropout
        '''
        var_mean = torch.var_mean(channel_matrix, dim=0) # tuple(var,mean)
        if train_dist:
            self.training_distribution = var_mean
        else:
            self.shifted_distribution = var_mean

    def get_mask(self, verbose = True):
        '''
        Retrieves mask for dropout
        '''

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
        '''
        forwards
        '''
        if self.training:
            mask = self.get_mask()
            print(x.shape)
            x = x * mask
        else:
            #print(self.recent_mean_diff.float())
            x = x * (1 - self.recent_mean_diff.float())
        return x

class DimensionIncreaser(nn.Module):
    '''
    increases complexity of the problem
    '''
    def __init__(self, input_dim, output_dim, use_activation=True):
        super(DimensionIncreaser, self).__init__()
        self.scramble_layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.activation = nn.Sigmoid()
        self.use_activation = use_activation
        self.requires_grad_(requires_grad=False)

    def forward(self, x):
        '''
        forwards
        '''
        x = self.scramble_layer(x)
        if self.use_activation:
            x = self.activation(x)
        return x
