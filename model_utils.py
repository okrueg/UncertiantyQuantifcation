from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import torch.utils
from torch.utils.data import DataLoader
from datasets import loadData
from model_architectures import BasicCNN


def train_fas_mnist(model: BasicCNN,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    test_loader: DataLoader,
                    num_epochs: int,
                    lr = 0.015,
                    save = False,
                    save_mode = 'loss',
                    verbose = True):
    '''
    model training sequence
    '''
    print(f'Used Device: {DEVICE}') if verbose else None
    model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=lr, momentum= 0.95) # , nesterov= True
    #scheduler = ExponentialLR(optimizer,0.90)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)

    best_val_loss = np.inf
    best_test_acc = -1 * np.inf

    best_model = deepcopy(model)
    train_losses = []
    val_losses = []

    train_accs = []
    test_accs = []

    #-----Train ----
    for epoch in range(num_epochs):
        #----Train Model ----
        model.train()

        overall_training_loss = 0
        overall_val_loss = 0
        test_loss = -1

        train_total_acc = torch.empty(len(train_loader))
        val_total_acc = torch.empty(len(val_loader))
        test_acc = -1

        train_output_scalar = torch.empty(len(train_loader))

        for batch_idx, (x, y) in tqdm(enumerate(train_loader),
                                           total=len(train_loader),
                                           desc=f'Epoch [{epoch+1}/{num_epochs}]: ',
                                           colour= 'green',
                                           disable= not verbose):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            optimizer.zero_grad()
            #----Run pre-forward pass to collect output for dropout----
            if hasattr(model.dropout, "weight_matrix"):
                with torch.no_grad():
                    model.eval()
                    first_output = model(x)

                model.dropout.weight_matrix = model.fc3.weight
                model.dropout.prev_output = first_output

            model.train()

            train_output = model(x,y)

            loss = loss_fn(train_output, y)

            overall_training_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.1)

            optimizer.step()

            # ---- Calculate train accuracys for the Batch ----
            predictions = torch.max(train_output, dim=1)[1]
            acc = torch.eq(y, predictions).int()

            train_total_acc[batch_idx] = torch.mean(acc, dtype= torch.float)

            u = torch.nn.functional.log_softmax(train_output.to('cpu').detach(), dim=-1)
            train_output_scalar[batch_idx] = torch.sum(abs(u), dim=1)[1]

        if hasattr(model.dropout, "drop_handeler"):
            #print(model.dropout.drop_handeler)
            model.dropout.drop_handeler.add_epoch()

        #---- VALIDATION -----
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                val_output = model(x)
                overall_val_loss += loss_fn(val_output, y).item()

                #---- Calculate train accuracys for the Batch ----
                predictions = torch.max(val_output, dim=1)[1]
                acc = torch.eq(y, predictions).int()

                val_total_acc[batch_idx] = torch.mean(acc, dtype= torch.float)

        test_stats = test_fas_mnist(model=model,
                            test_loader=test_loader,
                            verbose=False)

        test_loss, test_acc = test_stats[0], test_stats[1]

        overall_training_loss = overall_training_loss/len(train_loader)
        overall_val_loss = overall_val_loss/len(val_loader)

        train_total_acc = torch.mean(train_total_acc)
        val_total_acc = torch.mean(val_total_acc)

        train_accs.append(train_total_acc.item())
        test_accs.append(test_acc)

        train_losses.append(overall_training_loss)
        val_losses.append(overall_val_loss)

        train_output_scalar= torch.mean(train_output_scalar)

        scheduler.step()

        if verbose:
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.4f}')
            print(f'Train_output_scalar {train_output_scalar:.4f}')
            print(f'Training Loss:   {overall_training_loss:.4f} | Training Accuracy:   {train_total_acc:.4f}')
            print()
            print(f'Validation Loss: {overall_val_loss:.4f} | Validation Accuracy: {val_total_acc:.4f}')
            print()
            print(f'Testing Loss:    {test_loss:.4f} | Testing Accuracy:   {test_acc:.4f}')
            print()

        if save:
            if save_mode == 'loss':
                if  overall_val_loss < best_val_loss:
                    print(f"Saving new best validation loss: {overall_val_loss:.4f} < {best_val_loss:.4f}") if verbose else None


                    best_val_loss = overall_val_loss

                    best_model  = deepcopy(model)

            elif save_mode == 'accuracy':
                if test_acc > best_test_acc:
                    print(f"Saving new best testing accuracy: {test_acc:.4f} > {best_test_acc:.4f}") if verbose else None

                    best_test_acc = test_acc

                    best_model  = deepcopy(model)
            else:
                raise ValueError(f"Invalid save mode: {save_mode}")


            #torch.save(model.state_dict(), "model_best_val.path")
    return (train_losses, val_losses), (train_accs, test_accs), best_model

def test_fas_mnist(model: BasicCNN, test_loader: DataLoader, verbose = True):
    '''
    model testing functionality
    '''
    model.eval()
    with torch.no_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        label_acc = [[] for x in range(10)]
        overall_test_loss = 0

        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            test_output = model(x)
            overall_test_loss += loss_fn(test_output, y).item()

            #---- Calculate accuracys for the Batch ----
            predictions = torch.max(test_output, dim=1)[1]
            acc = torch.eq(y, predictions).int()

            # TODO: You could also sum for accuracy
            for ind, label in enumerate(y):
                label_acc[label].append(acc[ind].item())

        #---- Final calculation for returned Values ----
        overall_test_loss = overall_test_loss/len(test_loader)

        for ind, x in enumerate(label_acc):
            label_acc[ind] = sum(x)/len(x)

        total_acc = float(sum(label_acc)/10)

    if verbose:
        print(f"Testing Loss: {overall_test_loss}")
        print(f"Accuracies: {label_acc}")
        print(f"Total Accuracy: {total_acc:.4f}")
        print()


    return overall_test_loss, total_acc, label_acc

def model_grid_generator(x_range: tuple, y_range: tuple, grid_size: tuple):
    '''
    Generates a mesh grid of model param combinations
    '''
    x_range = np.linspace(start=x_range[0],
                        stop=x_range[1],
                        num=grid_size[0])

    y_range = np.linspace(start=y_range[0],
                            stop=y_range[1],
                            num=grid_size[1])

    x, y = np.meshgrid(x_range, y_range, indexing='xy')

    stacked_data = np.stack((x, y), axis=-1)

    x, y = np.array_split(stacked_data, 2, axis=-1)

    x = x.squeeze()
    y = y.squeeze()


    print(f"x results:\n{x.shape}")
    print(f"y result:\n{y.shape}")

    np.savetxt("x.csv", x, fmt='%.5f', delimiter=', ', header='x')
    np.savetxt("y.csv", y, fmt='%.5f', delimiter=', ', header='y')

    return stacked_data


def model_grid_training(model_params: np.ndarray,
                        model_changes: tuple,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        test_loader: DataLoader,
                        num_epochs: int, rank: int):
    '''
    Given a chunk of model parameters, this will train the model on each chunk
    '''
    
    model_options = {
        'num_classes': 10,
        'in_channels': 3,
        'out_feature_size': 2048,
        'use_reg_dropout': False,
        'dropout_prob': 0.5,
        'num_drop_channels': 2,
        'drop_certainty': 1,
    }

    def encompassed(params):

        model_options[model_changes[0]] = params[0]
        model_options[model_changes[1]] = params[1]

        print(model_params, model_options)

        verboscity = False
        # if rank == 0:
        #     verboscity = True

        model = BasicCNN(**model_options)
        _, min_training_loss, best_model = train_fas_mnist(model=model,
                                                      train_loader=train_loader,
                                                      val_loader=val_loader,
                                                      test_loader= test_loader,
                                                      num_epochs=num_epochs,
                                                      save=True,
                                                      save_mode='accuracy',
                                                      verbose=verboscity)

        min_training_loss= min(min_training_loss[0])

        test_loss, test_acc, _ = test_fas_mnist(model=best_model,test_loader=test_loader, verbose=verboscity)

        print(f'{rank} finished {params}')

        return np.array([min_training_loss, test_acc])


    test = np.apply_along_axis(encompassed, axis=-1, arr=model_params)
    return test


DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

#------- TEST the MODEL PROCESSCESS ------
# train_loader,val_loader,test_loader = loadData('CIFAR-10',batch_size= 200)

# model = BasicCNN(10,3,2048,0.5)

# _,_,model = train_fas_mnist(model=model,
#                             train_loader=train_loader,
#                             val_loader=val_loader,
#                             test_loader=test_loader,
#                             num_epochs=50,
#                             save=True,
#                             save_mode='accuracy')


# # model.load_state_dict(torch.load("model_best_val.path"), strict=False)
# print("new")
# test_fas_mnist(model, test_loader=test_loader)
