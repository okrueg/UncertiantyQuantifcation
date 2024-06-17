import random
import torch
import numpy as np
import torch.nn as nn
import torch.utils
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.functional import rotate, gaussian_blur
from torchvision.transforms import ElasticTransform

from model_architectures import basic_nn, dimension_increaser, basic_cnn
from data_set_generation import generate_data, plot_data_static
from torch.utils.data.sampler import  SubsetRandomSampler  #for validation test

class SubClassTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transform1 = rotate
        self.transform2 = ElasticTransform()
        self.transform3 = gaussian_blur
    def forward(self, img, is_subClass):

        if is_subClass == 1:
           img =  self.transform1(img, angle = 45)
           img = self.transform2(img)
           img = self.transform3(img, kernel_size=3,sigma=2)
           return img
        
        else:
            return img


class CustomFashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, label, label_percent, train=True, subclass_transform = None, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.additional_items = self.generate_subclass_labels(label,label_percent)
        self.subclass_transform = subclass_transform

    def generate_subclass_labels(self, class_num, class_percent):

        subclass_labels = torch.zeros_like(self.targets)

        # Find indices of all samples with class number
        class_ind = (self.targets == class_num).nonzero(as_tuple=False).float()

        sub_class_ind = class_ind[torch.rand_like(class_ind) <= class_percent].long()

        subclass_labels[sub_class_ind] = 1

        return subclass_labels

    def __getitem__(self, index):

        img, target = super().__getitem__(index)
        
        # Get the subclass_label for the current index
        subclass_label = self.additional_items[index]

        img = self.subclass_transform(img, subclass_label)
        
        
        return img, target, subclass_label

def train_fas_mnist(model: basic_cnn,
                    num_epochs,
                    lr = 0.1):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum= 0.9)
    scheduler = ExponentialLR(optimizer,0.95)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.75)

    best_val_loss = np.inf
    best_epoch = -1
    train_losses = []
    val_losses = []
    # TODO: Initializing a variable without datatype (this case dict?) is very bad code style!
    drop_info_epochs = None

    #-----Train ----
    for epoch in range(num_epochs):
        #----Train Model
        model.train()
        #print(epoch,model.special_dropout.engaged)
        overall_loss = 0
        val_loss = 0
        #print("trainloader length: ", len(trainloader))
        for batch_idx, (x, y, sub) in enumerate(trainloader):

            optimizer.zero_grad()

            # TODO: This is only required with the new dropout implementation
            with torch.no_grad():
                model.eval()

                first_output = model(x)
            model.train()

            model.dropout.weight_matrix = model.fc3.weight
            model.dropout.prev_output = first_output

            output = model(x,y)

            loss = loss_fn(output, y)
            
            overall_loss += loss.item()
            
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, sub) in enumerate(validloader):
                val_output = model(x)
                val_loss += loss_fn(val_output, y)

            # TODO: you only add the last loss value of this epoch
            train_losses.append(loss.item())
            val_losses.append(val_loss.item()/len(validloader))

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print(f'Validation Loss: {val_loss.item()/len(validloader):.4f}')

            print()
            test_fas_mnist(model)

        if drop_info_epochs == None:
            drop_info_epochs = model.dropout.info.copy()
        else:
            for item in model.dropout.info:
                # TODO: Are you sure you don't need the copy statement here?
                drop_info_epochs[item] = torch.vstack((drop_info_epochs[item], model.dropout.info[item]))            
            
        model.dropout.info = None


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"model_best_val.path")

        scheduler.step()
    drop_info = seperate_dropout_info(drop_info_epochs)
    return train_losses, val_losses, drop_info

def test_fas_mnist(model):
    model.eval()
    with torch.no_grad():
        label_acc = [[] for x in range(10)]

        for batch_idx, (x, y, sub) in enumerate(testloader):
            output = model(x)
            predictions = torch.max(output, dim=1)[1]
            acc = torch.eq(y, predictions).int()

            # TODO: You could also sum for accuracy
            for ind, label in enumerate(y):
                label_acc[label].append(acc[ind].item())

        for ind, x in enumerate(label_acc):
            label_acc[ind] = sum(x)/len(x)

        total_acc = sum(label_acc)/10

        print("Accuracies: ",label_acc)
        print("Total Accuracy", total_acc)
        print()

    return label_acc, total_acc

def seperate_dropout_info(drop_info_epochs):
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

batch_size = 200

#Define a transform to convert to images to tensor and normalize
transform = v2.Compose([v2.ToTensor(),
                            v2.Normalize((0.5,),(0.5,),)]) #mean and std have to be sequences (e.g., tuples), 
                                                                    # therefore we should add a comma after the values
                        
#Load the data: train and test sets
trainset = CustomFashionMNIST('~/.pytorch/F_MNIST_data',label=0, label_percent=0.00, download=True, train=True, transform=transform, subclass_transform= SubClassTransform())
testset = CustomFashionMNIST('~/.pytorch/F_MNIST_data',label=0, label_percent= 0.00, download=True, train=False, transform=transform, subclass_transform= SubClassTransform())

#Preparing for validaion test
indices = list(range(len(trainset)))
np.random.shuffle(indices)

#to get 80 20 split
split = int(np.floor(0.8 * len(trainset)))

print(len(indices[:split]), len(indices[split:]))

train_sample = SubsetRandomSampler(indices[:split])
valid_sample = SubsetRandomSampler(indices[split:])

#Data Loader
trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=batch_size)
validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

# model = basic_cnn(10)
# train_fas_mnist(model=model,
#                 num_epochs=10)


# #model = basic_cnn(10)
# model.load_state_dict(torch.load("model_best_val.path"))
# test_fas_mnist(model)