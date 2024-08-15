'''
'''
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from datasets import sparse2coarse

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


    def forward(self, x, y=None):
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
    def __init__(self, num_classes: int, in_channels: int,  out_feature_size: int):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=5, padding=2)

        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv2 = nn.Conv2d(96, 128, kernel_size=5, padding=2)

        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)

        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2304, 1024)

        self.fc2 = nn.Linear(1024, out_feature_size)

        self.fc3 = nn.Linear(out_feature_size, num_classes)

        self.ReLU = nn.LeakyReLU(0.2)
        #self.softmax = nn.Softmax(dim=1)

        self.activations = torch.empty(0)
        self._max_norm_val = 3
        self._eps = 1e-8

    def init_dropout(self, use_reg_dropout: bool, use_activations: bool, continous_dropout:bool, original_method: bool,
                      dropout_prob: float, num_drop_channels: int,  drop_certainty: float, drop_handler=None):
        
        self.use_reg_dropout = use_reg_dropout
        self.use_activations = use_activations
        self.originalMethod = original_method

        if use_reg_dropout:
            self.dropout = nn.Dropout(dropout_prob)

        else:
            self.dropout= ConfusionDropout(use_activations, original_method, continous_dropout, dropout_prob, int(num_drop_channels), drop_certainty, drop_handler)


    def forward(self, x: torch.Tensor, y=None):
        '''
        forwards through model
        '''
        assert x.isnan().any().item() is False
        assert self.dropout is not None

        x = self.ReLU(self.conv1(x))

        x = self.pool1(x)

        x = self.ReLU(self.conv2(x))

        x = self.pool2(x)

        x = self.ReLU(self.conv3(x))

        x = self.pool3(x)

        x = self.flatten(x)

        self.fc1.weight.data = self._max_norm(self.fc1.weight.data)
        x = self.ReLU(self.fc1(x))

        #x = self.dropout_reg(x)

        self.fc2.weight.data = self._max_norm(self.fc2.weight.data)
        x = self.ReLU(self.fc2(x))

        activations = x

        if isinstance(self.dropout, ConfusionDropout):
            x = self.dropout(x, y)
        else:
            x = self.dropout(x)

        self.fc3.weight.data = self._max_norm(self.fc3.weight.data)
        x = self.fc3(x)

        return x

    #https://github.com/kevinzakka/pytorch-goodies#max-norm-constraint
    def _max_norm(self, w):
        norm = w.norm(3, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))


class FromBayesCNN(nn.Module):
    '''
    a the deterministec version of the bays arch
    '''
    def __init__(self, num_classes: int, in_channels: int,  out_feature_size: int):
        super(FromBayesCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(12544, 2048)

        self.fc2 = nn.Linear(2048, 10)

        self.activation = nn.LeakyReLU()


    def init_dropout(self, use_reg_dropout: bool, use_activations: bool, continous_dropout:bool, original_method: bool,
                    dropout_prob: float, num_drop_channels: int,  drop_certainty: float):
    
        self.use_reg_dropout = use_reg_dropout
        self.use_activations = use_activations
        self.originalMethod = original_method

        if use_reg_dropout:
            self.dropout = nn.Dropout(dropout_prob)

        else:
            self.dropout= ConfusionDropout(use_activations, original_method, continous_dropout, dropout_prob, int(num_drop_channels), drop_certainty)


    def forward(self, x: torch.Tensor, y=None):
        '''
        forwards through model
        '''
        assert x.isnan().any().item() is False

        x = self.activation(self.conv1(x))

        x = self.activation(self.conv2(x))

        x = self.pool(x)

        x = self.flatten(x)

        x = self.activation(self.fc1(x))

        self.activations = x

        if hasattr(self, 'dropout'):
            if isinstance(self.dropout, ConfusionDropout):
                x = self.dropout(x, y)
            else:
                x = self.dropout(x)

        x = self.fc2(x)

        return x





class ConfusionDropout(nn.Module):
    '''
    A special form of dropout to challenge the model by causing class confusion
    '''
    def __init__(self, use_activations: bool, original_method: bool, continous_dropout: bool,
                  drop_percent: float, num_top_channels: int, drop_certianty: float, drop_handler = None):
      
        super().__init__()

        self.weight_matrix = torch.empty(0)
        self.prev_output = torch.empty(0)

        self.drop_percent = drop_percent
        self.num_top_channels = num_top_channels
        self.drop_certianty = drop_certianty
        self.drop_handeler = drop_handler

        self.original_method = original_method
        self.use_activations = use_activations
        self.continous_dropout = continous_dropout

    def get_mask(self, x: torch.Tensor, y=None):
        if y is None:
            y = self.y
        '''
        retrieves mask for doc string
        '''
        
        #retrieve the indicies of the 2 highest model predictions
        top_ind = torch.topk(self.prev_output, k= self.num_top_channels, dim=1)[1]

        #print(top_ind)

        #select the weights associated with those model predictions
        selected_weight_cols = self.weight_matrix[top_ind] #Shape: batch, 2, feature size

        if self.original_method:

            selected_weight_diffs = selected_weight_cols[:, 0, :] - selected_weight_cols[:, 1, :] #Shape: batch, feature size

            #weight * channel, Higher Score means higherdrop rate
            #print(x.shape, selected_weight_diffs.shape, self.weight_matrix.shape)

            if self.use_activations:
                stored_scores = x * selected_weight_diffs
                
            else:
                stored_scores = selected_weight_diffs

            scores = abs(stored_scores)
        
        else:
            selected_weight_diffs = selected_weight_cols - self.weight_matrix[y].unsqueeze(1)

            # BUG Test
            #print(torch.mean(x.var(dim=1)))
            if self.use_activations:
                stored_scores = x.unsqueeze(1) * selected_weight_diffs
            else:
                stored_scores = selected_weight_diffs

            scores = abs(stored_scores)

            scores = torch.max(scores, dim=1)[0]

            # FIX
            stored_scores = scores

        if not self.continous_dropout:
            #Number of channels to drop
            num_dropped = int(x.shape[1] * self.drop_percent)

            #select num_dropped num channels with the highest score
            dropped_channels = torch.topk(scores, k=num_dropped, dim=1, largest=True)[1]

            mask = torch.ones_like(x).bool()

            # values above certianty set to false IE with certainty of 1.0 this is the same as = False
            certianty_mask = torch.rand((mask.shape[0], num_dropped), device=mask.device) > self.drop_certianty      

            batch_indices = torch.arange(mask.shape[0]).unsqueeze(1).repeat(1, num_dropped)

            mask[batch_indices, dropped_channels] = certianty_mask
        
        else:

            ind = torch.argsort(scores, dim=1, descending=False)

            ind = ind.to(device=x.device)

            ranks = torch.zeros_like(ind, dtype=torch.int64, device=x.device)
    
            batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).repeat(1, x.shape[1])

            ranks[batch_indices, ind] = torch.arange(1, x.shape[1] + 1, device=x.device)

            ranks = ranks/x.shape[1]

            # print('scores', scores[0])

            # print('Before', ranks[0])

            ranks = self.contFunc(ranks, mid= (1.05- self.drop_percent))
           
            mask = torch.ones_like(x).bool()
            
            mask = torch.rand_like(x, device=mask.device) >= (ranks)

            # print('After',ranks[0])
            # print('mask', mask[0])

            #print(torch.count_nonzero(mask)/(10*200))



        if self.drop_handeler is not None:
            self.drop_handeler.store_forwardpass(y.to('cpu').detach(),
                                                    mask.to('cpu').detach(),
                                                    top_ind.to('cpu').detach(),
                                                    stored_scores.to('cpu').detach())

        return mask


    def contFunc(self, ranks: torch.Tensor, mid=0.5, slope=2):
        '''
        applys a smoothing Tanh function to the ranks
        the mid will modify the halfway of the tanHfunc (where x ==y)
        Higher mid is less dropped
        slope alters the amount of smoothing, a higher slope makes the decision more Binary

        https://www.desmos.com/calculator/q9wmfawwqz
        '''

        func_ranks = slope * (ranks - mid)
        func_ranks = torch.tanh(func_ranks)

        return (func_ranks+1)/2


    def forward(self, x: torch.Tensor, y = None):
        '''
        forwards dropout
        '''
        if self.training:
            if self.prev_output is not None:
                mask = self.get_mask(x, y)
                x = x * mask
        else:
                x = x * (1 - (self.drop_percent * self.drop_certianty)) # keep expected value
        return x

class SuperLabelDropout(ConfusionDropout):
    def __init__(self, drop_percent: float, num_top_channels: int):
        # TODO
        super().__init__(False, False, False, drop_percent, num_top_channels, drop_certianty=1, drop_handler= DropoutDataHandler())

    def get_mask(self, x, y):
        '''
        retrieves mask for doc string
        '''
        supers_y = sparse2coarse(y)

        # print(supers)

        #retrieve the indicies of the 2 highest model predictions
        top_ind = torch.topk(self.prev_output, k= self.num_top_channels, dim=1)[1]

        top_super_ind = sparse2coarse(top_ind, device='mps')



        #select the weights associated with those model predictions
        selected_weight_cols = self.weight_matrix[top_ind] #Shape: batch, 2, feature size

        test = False
        # method 1 distance to correct
        if test:
            selected_weight_diffs = selected_weight_cols - self.weight_matrix[y].unsqueeze(1) #Shape: batch, feature size

            scores = abs(x.unsqueeze(1) * selected_weight_diffs) #Shape: batch, feature

            scores = torch.mean(scores, dim=1)

        else:

            selected_weight_diffs = selected_weight_cols[:, 0, :] - selected_weight_cols[:, 1, :] #Shape: batch, feature size

            #weight * channel, Higher Score means higherdrop rate
            scores = abs(x * selected_weight_diffs)


        #Number of channels to drop
        num_dropped = int(x.shape[1] * self.drop_percent)

        #select num_dropped num channels with the highest score
        dropped_channels = torch.topk(scores, k=num_dropped, dim=1, largest=True)[1]

        mask = torch.ones_like(x).bool()

        # values above certianty set to false IE with certainty of 1.0 this is the same as = False
        certianty_mask = torch.rand((mask.shape[0], num_dropped), device=mask.device) > self.drop_certianty
        

        batch_indices = torch.arange(mask.shape[0]).unsqueeze(1).repeat(1, num_dropped)

        mask[batch_indices, dropped_channels] = certianty_mask

        if y is not None:
            self.drop_handeler.store_forwardpass(y.to('cpu').detach(),
                                                 mask.to('cpu').detach(),
                                                 top_ind.to('cpu').detach(),
                                                 scores.to('cpu').detach())

        return mask

class DropoutDataHandler():
    '''
    for collecting info in our special dropout
    '''
    def __init__(self):
        self.batch_info = {}
        self.epoch_info = {}
    
    def __str__(self) -> str:
        shapes = "DropoutDataHandler:\n"
        if self.epoch_info:
            for key, value in self.epoch_info.items():
                shapes += f'{key}: {value.shape}\n'

        return shapes


    def store_forwardpass(self, y, mask, top_ind, selected_weight_diffs):
        '''
        Store info about dropout's pass
        '''

        # Make epoch compatable
        y = torch.unsqueeze(y, dim=0)
        dropped_channels = torch.unsqueeze(mask, dim=0)
        top_ind = torch.unsqueeze(top_ind, dim=0)
        selected_weight_diffs = torch.unsqueeze(selected_weight_diffs, dim=0)

        if not self.batch_info: # if this is the first batch
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
        if not self.epoch_info:
            self.epoch_info = self.batch_info.copy()
        else:
            for item in self.batch_info:
                self.epoch_info[item] = torch.vstack((self.epoch_info[item], self.batch_info[item]))

        self.batch_info = {}


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
        label_indices = torch.nonzero((self.epoch_info["labels"][epoch] == label), as_tuple=True)[0]


        selected_per_label = self.epoch_info[inquiry][epoch][label_indices, :]



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
            #print(f'x shape: {x.shape}',f'mask shape: {mask.shape}')
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
