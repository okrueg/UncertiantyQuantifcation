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

from model_architectures import basic_nn, dimension_increaser, basic_cnn
from data_set_generation import generate_data, plot_data_static
from torch.utils.data.sampler import  SubsetRandomSampler  #for validation test

class MyCustomTransform(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
    def forward(self, img, is_subClass):

        if is_subClass == 1:
           return self.transform(img, angle = 45)
        
        else:
            return img


class CustomFashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, train=True, subclass_transform = None, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.additional_items = self.generate_subclass_labels(0,0.5)
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

        self.subclass_transform(img, subclass_label)
        
        
        return img, target, subclass_label

def train_fas_mnist(num_epochs = 25,
                    lr = 0.1):
    model = basic_cnn(10)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum= 0.9)
    scheduler = ExponentialLR(optimizer,0.95)

    best_val_loss = np.inf
    best_epoch = -1
    train_losses = []
    val_losses = []

    #-----Train ----
    for epoch in range(num_epochs):
        #----Train Model
        model.train()
        overall_loss = 0
        val_loss = 0
        for batch_idx, (x, y, sub) in enumerate(trainloader):

            optimizer.zero_grad()
            #x = dim_inc(x)
            output = model(x)

            loss = loss_fn(output, y)
            
            overall_loss += loss.item()
            
            loss.backward()

            optimizer.step()
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, sub) in enumerate(validloader):
                val_output = model(x)
                val_loss += loss_fn(val_output, y)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Validation Loss: {val_loss.item()/len(validloader):.4f}')
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"model_best_val.path")

        scheduler.step()


def test_fas_mnist(model):
    model.eval()
    with torch.no_grad():
        label_acc = [[] for x in range(10)]

        for batch_idx, (x, y, sub) in enumerate(testloader):
            output = model(x)
            predictions = torch.max(output, dim=1)[1]
            acc = torch.eq(y, predictions).int()

            interest_class = predictions[y == 0]
            sub = sub[y==0]
            regular = interest_class[sub == 0]
            other = interest_class[sub == 1]
            #wrong
            print(torch.nonzero(regular.float()).shape[0]/regular.shape[0], torch.nonzero(other.float()).shape[0]/other.shape[0])



            for ind, label in enumerate(y):
                label_acc[label].append(acc[ind].item())

        for ind, x in enumerate(label_acc):
            label_acc[ind] = sum(x)/len(x)
        
        print(label_acc)
        print(sum(label_acc)/10)

#def train_fas_mnist():


batch_size = 200

#Define a transform to convert to images to tensor and normalize
transform = v2.Compose([v2.ToTensor(),
                            v2.Normalize((0.5,),(0.5,),)]) #mean and std have to be sequences (e.g., tuples), 
                                                                    # therefore we should add a comma after the values
                        
#Load the data: train and test sets
trainset = CustomFashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=transform, subclass_transform= MyCustomTransform(rotate))
testset = CustomFashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=False, transform=transform, subclass_transform= MyCustomTransform(rotate))

#Preparing for validaion test
indices = list(range(len(trainset)))
np.random.shuffle(indices)
#to get 20% of the train set
split = int(np.floor(0.2 * len(trainset)))
train_sample = SubsetRandomSampler(indices[:split])
valid_sample = SubsetRandomSampler(indices[split:])

#Data Loader
trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=batch_size)
validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

#train_fas_mnist()


model = basic_cnn(10)
model.load_state_dict(torch.load("model_best_val.path"))
test_fas_mnist(model)