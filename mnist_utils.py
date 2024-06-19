'''
TBD
'''
from copy import deepcopy
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data.sampler import SubsetRandomSampler #for validation test

from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.functional import rotate, gaussian_blur
from torchvision.transforms import ElasticTransform

from model_architectures import BasicCNN


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


def train_fas_mnist(model: BasicCNN,
                    num_epochs: int,
                    lr = 0.1,
                    save = False,
                    save_mode = 'loss',
                    verbose = True):
    '''
    model training sequence
    '''
    print(f'Used Device: {device}') if verbose else None
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum= 0.9, nesterov= True)
    scheduler = ExponentialLR(optimizer,0.90)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.75)

    best_val_loss = np.inf
    best_test_acc = -1 * np.inf

    best_model = deepcopy(model)
    train_losses = []
    val_losses = []
    #-----Train ----
    for epoch in range(num_epochs):
        #----Train Model ----
        model.train()
        overall_loss = 0
        val_loss = 0
        for batch_idx, (x, y, sub) in tqdm(enumerate(trainloader),
                                           total=len(trainloader),
                                           desc=f'Epoch [{epoch+1}/{num_epochs}]: ',
                                           colour= 'green',
                                           disable= not verbose):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

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

        if hasattr(model.dropout, "drop_handeler"):
            model.dropout.drop_handeler.add_epoch()
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y, sub) in enumerate(validloader):
                x = x.to(device)
                y = y.to(device)

                val_output = model(x)
                val_loss += loss_fn(val_output, y)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item()/len(validloader))

        scheduler.step()
        test_acc = test_fas_mnist(model=model,
                                 verbose=False)[1]
        if verbose:
            print(f'Training Loss: {loss.item():.4f}')
            print(f'Validation Loss: {val_loss.item()/len(validloader):.4f}')
            print()
            print(f'Testing Accuracy: {test_acc:.4f}')
            print()

        if save:
            if save_mode == 'loss':
                if  val_loss < best_val_loss:
                    print(f"Saving new best validation loss: {val_loss:.4f} < {best_val_loss:.4f}") if verbose else None

                    best_val_loss = val_loss

                    best_model  = deepcopy(model)

            elif save_mode == 'accuracy':
                if test_acc > best_test_acc:
                    print(f"Saving new best testing accuracy: {test_acc:.4f} > {best_test_acc:.4f}") if verbose else None

                    best_test_acc = test_acc

                    best_model  = deepcopy(model)
            else:
                raise ValueError(f"Invalid save mode: {save_mode}")


            #torch.save(model.state_dict(), "model_best_val.path")
    return train_losses, val_losses, best_model


def test_fas_mnist(model: BasicCNN, verbose = True):
    '''
    model testing functionality
    '''
    model.eval()
    with torch.no_grad():
        label_acc = [[] for x in range(10)]

        for batch_idx, (x, y, sub) in enumerate(testloader):
            x = x.to(device)
            y = y.to(device)
        
            output = model(x)
            predictions = torch.max(output, dim=1)[1]
            acc = torch.eq(y, predictions).int()

            for ind, label in enumerate(y):
                label_acc[label].append(acc[ind].item())

        for ind, x in enumerate(label_acc):
            label_acc[ind] = sum(x)/len(x)

        total_acc = sum(label_acc)/10

    if verbose:
        print(f"Accuracies: {label_acc}")
        print(f"Total Accuracy: {total_acc:.4f}")
        print()

    return label_acc, total_acc


def model_grid_generator(drop_prob_range: tuple, feature_size_range: tuple, num: int):
    '''
    Generates a mesh grid of model param combinations
    '''
    drop_prob_range = np.linspace(start=drop_prob_range[0],
                        stop=drop_prob_range[1],
                        num=num)

    feature_size_range = np.linspace(start=feature_size_range[0],
                            stop=feature_size_range[1],
                            num=num)

    x, y = np.meshgrid(drop_prob_range, feature_size_range, indexing='xy')

    stacked_data = np.stack((x, y), axis=-1)

    drop, feature = np.array_split(stacked_data, 2, axis=-1)

    drop = drop.squeeze()
    feature = feature.squeeze()


    print(f"Drop results:\n{drop.shape}")
    print(f"feature result:\n{feature.shape}")

    np.savetxt("drop.csv", drop, fmt='%.5f', delimiter=', ', header='drop')
    np.savetxt("feature.csv", feature, fmt='%.5f', delimiter=', ', header='Feature')

    return stacked_data

def model_grid_training(model_params: np.ndarray, num_epochs: int, rank: int):
    '''
    Given a chunk of model parameters, this will train the model on each chunk
    '''

    def encompassed(model_args: np.ndarray):
        verboscity = False
        if rank == 0:
            print(model_args[0], model_args[1])
            verboscity = True

        model = BasicCNN(num_classes=10,
                         feature_size=int(model_args[1]),
                         dropout_prob=model_args[0])
        _, min_val_loss, best_model = train_fas_mnist(model=model,
                                                        num_epochs=num_epochs,
                                                        save=True,
                                                        save_mode='accuracy',
                                                        verbose=verboscity)

        min_val_loss= min(min_val_loss)

        _, total_acc = test_fas_mnist(model=best_model, verbose=False)

        return [min_val_loss, total_acc]


    test = np.apply_along_axis(encompassed, axis=-1, arr=model_params)
    return test

BATCH_SIZE = 400

device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

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

#Preparing for validaion test
indices = list(range(len(trainset)))
np.random.shuffle(indices)

#to get 80 20 split
split = int(np.floor(0.8 * len(trainset)))

#print(len(indices[:split]), len(indices[split:]))

train_sample = SubsetRandomSampler(indices[:split])
valid_sample = SubsetRandomSampler(indices[split:])

#Data Loader
trainloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=BATCH_SIZE)
validloader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=BATCH_SIZE)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)



#------- TEST the MODEL PROCESSCESS ------
# model = BasicCNN(10,512,0.75)
# _,_,model = train_fas_mnist(model=model,
#                             num_epochs=15,
#                             save=True,
#                             save_mode='accuracy')


# # model.load_state_dict(torch.load("model_best_val.path"), strict=False)
# print("new")
# test_fas_mnist(model)
