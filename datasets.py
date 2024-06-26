'''
TBD
'''
import torch
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.sampler import SubsetRandomSampler #for validation test

from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.functional import rotate, gaussian_blur
from torchvision.transforms import ElasticTransform



class SubClassTransform(torch.nn.Module):
    '''
    Modifys a certian sub population of a label
    '''
    def __init__(self):
        super().__init__()
        self.transform1 = rotate
        self.transform2 = ElasticTransform()
        self.transform3 = gaussian_blur
    def forward(self, img, is_sub_class):
        '''
        modufys images of subclass
        '''
        if is_sub_class == 1:
            img =  self.transform1(img, angle = 45)
            img = self.transform2(img)
            img = self.transform3(img, kernel_size=3,sigma=2)
            return img

        else:
            return img


class CustomFashionMNIST(datasets.FashionMNIST):
    '''
    Capable of creating subclass images in dataset
    '''
    def __init__(self, root, label, label_percent,
                 train=True,
                 subclass_transform = None,
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.additional_items = self.generate_subclass_labels(label,label_percent)
        self.subclass_transform = subclass_transform

    def generate_subclass_labels(self, class_num, class_percent):
        '''
        generates a tensor populated with the labels of the subclasses
        '''

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

def loadData(dataset: str, batch_size: int, train_val_split = 0.99):

    match dataset:
        case 'MNIST':

            #Define a transform to convert to images to tensor and normalize
            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                    v2.Normalize((0.5,),(0.5,),)]) #mean and std have to be sequences (e.g., tuples)

            #Load the data: train and test sets
            trainset = CustomFashionMNIST('~/.pytorch/F_MNIST_data',
                                        label=0, label_percent=0.00,
                                        download=True,
                                        train=True,
                                        transform=transform, subclass_transform= SubClassTransform())
            testset = CustomFashionMNIST('~/.pytorch/F_MNIST_data',
                                        label=0, label_percent= 0.00,
                                        download=True,
                                        train=False,
                                        transform=transform, subclass_transform= SubClassTransform())
        case 'CIFAR-10':

            transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                    v2.Normalize((0.5,),(0.5,),)]) #mean and std have to be sequences (e.g., tuples)
            
            trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    #Preparing for validaion test
    indices = list(range(len(trainset)))
    np.random.shuffle(indices)

    #to get 80 20 split
    split = int(np.floor(train_val_split * len(trainset)))

    #print(len(indices[:split]), len(indices[split:]))

    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[split:])

    #Data Loader
    trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, validloader, testloader
