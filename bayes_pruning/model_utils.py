from copy import copy, deepcopy

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import plotly.express as px
import torch.utils
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError
from model_architectures import BasicCNN
from wideresnet import WideResNet

DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'

class ActivationLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, gamma: float, label_smoothing: float):
        super(ActivationLoss, self).__init__(label_smoothing=label_smoothing)
        
        self.gamma = gamma
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, activations: torch.Tensor) -> torch.Tensor:

        if activations is None:
            return super().forward(input, target)

        #activation_norm = torch.sqrt(torch.sum(torch.square(activations)))
        activation_norm = torch.norm(activations, p=2)
        #print(activation_norm)
        #print((self.gamma *activation_norm).item())


        return super().forward(input, target) + self.gamma * activation_norm

def train_fas_mnist(model: BasicCNN,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    test_loader: DataLoader,
                    num_epochs: int,
                    activation_gamma: float,
                    lr: float,
                    save = False,
                    save_mode = 'loss',
                    verbose = True):
    '''
    model training sequence
    '''
    print(f'Used Device: {DEVICE}') if verbose else None
    model.to(DEVICE)

    #loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn = ActivationLoss(gamma=activation_gamma, label_smoothing=0.1)

    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma=0.90)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*num_epochs), int(0.8*num_epochs)], gamma=0.2)

    best_val_loss = np.inf
    best_test_acc = -1 * np.inf

    best_model = deepcopy(model)
    train_losses = []
    val_losses = []

    train_accs = []
    test_accs = []

   #train_calibs = []
    test_calibs = []

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
            if hasattr(model, "dropout"):
                if hasattr(model.dropout, "weight_matrix"):
                    with torch.no_grad():
                        model.eval()
                        first_output = model(x)

                    model.dropout.weight_matrix = model.fc3.weight
                    model.dropout.prev_output = first_output

                    # BUG Test
                    #print((model.dropout.weight_matrix).var(dim=0).shape, model.dropout.weight_matrix.shape )

            model.train()

            train_output = model(x,y)

            activations = None
            if hasattr(model, 'activations'):
                activations = model.activations


            loss = loss_fn(train_output, y, activations)

            overall_training_loss += loss.item()

            loss.backward()
            
            # min_grad = float('inf')
            # max_grad = float('-inf')

            # for param in model.parameters():
            #     if param.grad is not None:
            #         min_grad = min(min_grad, param.grad.min().item())
            #         max_grad = max(max_grad, param.grad.max().item())

            # print(f'Smallest gradient: {min_grad}')
            # print(f'Largest gradient: {max_grad}')

            #nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

            optimizer.step()

            # ---- Calculate train accuracys for the Batch ----
            predictions = torch.max(train_output, dim=1)[1]

            #print(predictions)
            acc = torch.eq(y, predictions).int()

            train_total_acc[batch_idx] = torch.mean(acc, dtype= torch.float)

            u = torch.nn.functional.log_softmax(train_output.to('cpu').detach(), dim=-1)
            train_output_scalar[batch_idx] = torch.sum(abs(u), dim=1)[1]


        #print(model.dropout.drop_percent)
        if hasattr(model, "dropout"):
            if hasattr(model.dropout, "drop_handeler") and model.dropout.drop_handeler is not None:

                # BUG TESTING
                count_tensor = (model.dropout.drop_handeler.batch_info["selected_weight_indexs"] == model.dropout.drop_handeler.batch_info["labels"].unsqueeze(2)).sum(dim=2)
                print("Epoch mean: ", torch.mean(count_tensor.to(torch.float), dim=1)) if verbose else None

                #print(model.dropout.drop_handeler)
                model.dropout.drop_handeler.add_epoch()

        #---- VALIDATION -----
        model.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                val_output = model(x,y)
                overall_val_loss += loss_fn(val_output, y, None).item()

                #---- Calculate train accuracys for the Batch ----
                predictions = torch.max(val_output, dim=1)[1]
                acc = torch.eq(y, predictions).int()

                val_total_acc[batch_idx] = torch.mean(acc, dtype= torch.float)

        test_stats = test_fas_mnist(model=model,
                            test_loader=test_loader,
                            verbose=False)

        test_loss, test_calib_error, test_acc = test_stats[0],test_stats[1], test_stats[2]

        overall_training_loss = overall_training_loss/len(train_loader)
        overall_val_loss = overall_val_loss/len(val_loader)

        train_total_acc = torch.mean(train_total_acc)
        val_total_acc = torch.mean(val_total_acc)

        train_accs.append(train_total_acc.item())
        test_accs.append(test_acc)

        train_losses.append(overall_training_loss)
        val_losses.append(overall_val_loss)

        test_calibs.append(test_calib_error)

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
            print(f'Test Calibration error:  {test_calib_error:.4f}')
        save_name = ""
        if save:
            if type(model).__name__ == "BasicCNN":
                save_name = f'model_{num_epochs}_isreg_{model.use_reg_dropout}_useAct_{model.use_activations}_original_method_{model.originalMethod}.path'
            else:
                save_name = f'{type(model).__name__}_{num_epochs}.path'
            if save_mode == 'loss':
                if  overall_val_loss < best_val_loss:
                    print(f"Saving new best validation loss: {overall_val_loss:.4f} < {best_val_loss:.4f}") if verbose else None


                    best_val_loss = overall_val_loss

                    torch.save(model, save_name)

            elif save_mode == 'accuracy':
                if test_acc > best_test_acc:
                    print(f"Saving new best testing accuracy: {test_acc:.4f} > {best_test_acc:.4f}") if verbose else None

                    best_test_acc = test_acc

                    #best_model  = deepcopy(model)
                    torch.save(model, save_name)
            else:
                raise ValueError(f"Invalid save mode: {save_mode}")


    return (train_losses, val_losses), (train_accs, test_accs), save_name


def test_fas_mnist(model: BasicCNN, test_loader: DataLoader, evaluate= True, verbose = True):
    '''
    model testing functionality
    '''
    with torch.no_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        calib_metric = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')

        label_acc = [[] for x in range(10)]
        overall_test_loss = 0
        activations_ = []
        calibrations = []

        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            if hasattr(model, "dropout"):
                if hasattr(model.dropout, "weight_matrix"):
                    with torch.no_grad():
                        model.eval()
                        first_output = model(x)
                    model.dropout.weight_matrix = model.fc3.weight
                    model.dropout.prev_output = first_output

            if evaluate:
                model.eval()
            else:
                model.train()

            test_output = model(x, y)

            calibrations.append(calib_metric(test_output, y).item())


            #----BUG TEST-----
            #print('output',torch.softmax(test_output, dim=1)[0:2])


            overall_test_loss += loss_fn(test_output, y).item()

            #---- Calculate accuracys for the Batch ----
            predictions = torch.max(test_output, dim=1)[1]
            acc = torch.eq(y, predictions).int()

            # TODO: You could also sum for accuracy
            for ind, label in enumerate(y):
                label_acc[label].append(acc[ind].item())

        #---- Final calculation for returned Values ----
        #all_acts = torch.mean(torch.stack(activations_), dim=0)

        #---- Correlations
        # corr = feature_correlation(all_acts)

        # heatmap = px.imshow(corr, text_auto=False, color_continuous_midpoint= 0.0, color_continuous_scale= 'RdBu')#, color_continuous_midpoint= 0.0001)
        # heatmap.show()

        overall_test_loss = overall_test_loss/len(test_loader)

        for ind, x in enumerate(label_acc):
            label_acc[ind] = sum(x)/len(x)

        total_acc = float(sum(label_acc)/10)

        calib_error = sum(calibrations)/len(calibrations)

    if verbose:
        print(f"Testing Loss: {overall_test_loss}")
        print()
        print(f"Accuracies: {label_acc}")
        print(f"Total Accuracy: {total_acc:.4f}")
        print()
        print(f'Calibration error:{calib_error}')
        print()


    return overall_test_loss, calib_error, total_acc, label_acc

def test_survival(model: BasicCNN, test_loader: DataLoader, steps = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]):

    survival_accs = []

    for drop_rate in steps:
        model.init_dropout(use_reg_dropout=False,
                           use_activations=True,
                           continous_dropout=False,
                           original_method= True,
                           dropout_prob= drop_rate,
                           num_drop_channels=3,
                           drop_certainty=1.0)
        model.train()

        _,_, acc, _ = test_fas_mnist(model, test_loader, evaluate=False, verbose=False)
        survival_accs.append(acc)

    return survival_accs

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
    }
    dropout_options = {
        'use_reg_dropout': False,
        'use_activations': False,
        'original_method': False,
        'dropout_prob': 0.5,
        'num_drop_channels': 3,
        'drop_certainty': 1,
    }

    def encompassed(params):

        dropout_options[model_changes[0]] = params[0]
        dropout_options[model_changes[1]] = params[1]

        verboscity = False
        # if rank == 0:
        #     verboscity = True

        model = BasicCNN(**model_options)

        model.init_dropout(**dropout_options)

        _, min_training_loss, best_model = train_fas_mnist(model=model,
                                                      train_loader=train_loader,
                                                      val_loader=val_loader,
                                                      test_loader= test_loader,
                                                      num_epochs=num_epochs,
                                                      activation_gamma= 0.001,
                                                      lr = 0.001,
                                                      save=True,
                                                      save_mode='accuracy',
                                                      verbose=verboscity)

        min_training_loss= min(min_training_loss[0])

        test_loss,_, test_acc, _ = test_fas_mnist(model=best_model,test_loader=test_loader, verbose=verboscity)

        print(f'{rank} finished {params}')

        return np.array([min_training_loss, test_acc])


    test = np.apply_along_axis(encompassed, axis=-1, arr=model_params)
    return test

def feature_correlation(activations):

    # x.to('mps')
    # y.to('mps')
    # model.to('mps')

    # _, activations = model(x, y)
    
    activations = activations.to('cpu').numpy()

    print(activations.shape)
    
    return np.cov(activations, rowvar=False)


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

#------- Model Survival Testing

# regular_Model = torch.load("model_31_isreg_True_useAct_True_original_method_True.path")
# our_Model = torch.load("model_30_isreg_False_useAct_True_original_method_True.path")

# print(test_survival(regular_Model, test_loader=test_loader))

# print('Our Model')

# print(test_survival(our_Model, test_loader=test_loader))
