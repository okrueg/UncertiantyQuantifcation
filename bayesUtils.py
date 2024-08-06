from copy import deepcopy
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError
from bayesian_torch.models.dnn_to_bnn import get_kl_loss
import numpy as np
from bayesArchetectures import BNN
from tqdm import tqdm


from datasets import loadData

DEVICE = 'cpu'
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'

def train_Bayes(model: BNN,
                train_loader: DataLoader,
                test_loader: DataLoader,
                num_epochs: int,
                num_mc: int,
                temperature: float,
                lr: float,
                from_dnn = False,
                save = False,
                save_mode = 'loss',
                verbose = True):
    '''
    model training sequence
    '''
    print(f'Used Device: {DEVICE}') if verbose else None
    model.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss() #label_smoothing=0.1


    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 7, gamma=0.90)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*num_epochs), int(0.8*num_epochs)], gamma=0.2)

    best_test_acc = -1 * np.inf

    train_losses = []
    val_losses = []

    train_accs = []
    test_accs = []

    #-----Train --------------------------------
    for epoch in range(num_epochs):

        #----Train Model ----
        model.train()

        overall_training_loss = 0
        overall_val_loss = 0
        test_loss = -1

        train_total_acc = torch.empty(len(train_loader))

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
            if hasattr(model, 'dropout'):
                if hasattr(model.dropout, "weight_matrix"):
                    with torch.no_grad():
                        model.eval()
                        first_output, _ = model(x)

                    model.dropout.weight_matrix = model.fc3.weight
                    model.dropout.prev_output = first_output

            model.train()

            # compute output
            output_ = []
            kl_ = []

            if from_dnn:
                for mc_run in range(num_mc):
                    output = model(x, y)
                    output_.append(output)

                    kl = get_kl_loss(model)

                    kl_.append(kl)

            else:
                for mc_run in range(num_mc):

                    output, kl = model(x, y)

                    output_.append(output)

                    kl_.append(kl)

            train_output = torch.mean(torch.stack(output_), dim=0)

            kl = torch.mean(torch.stack(kl_), dim=0)

            cross_entropy_loss = loss_fn(train_output, y)

            scaled_kl = kl / x.shape[0]

            # Make sure loss isnt inf
            assert cross_entropy_loss != torch.inf
            assert scaled_kl != torch.inf, f'BAD KL LOSS: {scaled_kl}'

            #print(f'crossLoss: {cross_entropy_loss} KL Loss: {scaled_kl}')

            #ELBO loss
            loss = cross_entropy_loss + temperature * scaled_kl

            overall_training_loss += loss.item()

            loss.backward()

        #--------EXPLODING GRADIENT TESTING----------------
            # min_grad = float('inf')
            # max_grad = float('-inf')

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if abs(param.grad.min().item()) < min_grad:
            #             min_grad = abs(param.grad.min().item())
            #             print(f'New Min from {name} is {min_grad}')

            #         if abs(param.grad.max().item()) > max_grad:
            #             max_grad = abs(param.grad.max().item())
            #             print(f'New Max from {name} is {max_grad}')

            # print(f'Smallest gradient: {min_grad}')
            # print(f'Largest gradient: {max_grad}')

            #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5)

        #-------Exploding KL loss testing ------------------
            # for layer in model.modules():
            #     if hasattr(layer, "kl_loss"):
            #         sigma_weight = torch.log1p(torch.exp(layer.rho_kernel))
            #         # print(layer.rho_kernel.min())
            #         # print(sigma_weight.min())
            #         # print(torch.log(sigma_weight).min())

            #         kl = torch.log(layer.prior_weight_sigma) - torch.log(
            #             sigma_weight) + (sigma_weight**2 + (layer.mu_kernel - layer.prior_weight_mu)**2) / (2 * (layer.prior_weight_sigma**2)) - 0.5

            #         print(kl.mean())

            optimizer.step()


            # ---- Calculate train accuracys for the Batch ----
            predictions = torch.max(train_output, dim=1)[1]
            #print(predictions)
            acc = torch.eq(y, predictions).int()

            train_total_acc[batch_idx] = torch.mean(acc, dtype= torch.float)

            u = torch.nn.functional.log_softmax(train_output.to('cpu').detach(), dim=-1)
            train_output_scalar[batch_idx] = torch.sum(abs(u), dim=1)[1]


        if hasattr(model, 'dropout'):
            if hasattr(model.dropout, "drop_handeler") and model.dropout.drop_handeler is not None:

                # BUG TESTING
                # count_tensor = (model.dropout.drop_handeler.batch_info["selected_weight_indexs"] == model.dropout.drop_handeler.batch_info["labels"].unsqueeze(2)).sum(dim=2)
                # print("Epoch mean: ", torch.mean(count_tensor.to(torch.float), dim=1)) if verbose else None

                #print(model.dropout.drop_handeler)
                model.dropout.drop_handeler.add_epoch()

        overall_training_loss = overall_training_loss/len(train_loader)
        train_total_acc = torch.mean(train_total_acc)
        train_losses.append(overall_training_loss)
        train_accs.append(train_total_acc.item())

        test_stats = test_Bayes(model=model,
                    test_loader=test_loader,
                    num_mc=10,
                    from_dnn=from_dnn,
                    verbose=False)

        test_loss, test_acc = test_stats[0], test_stats[2]

        test_accs.append(test_acc)


        train_output_scalar= torch.mean(train_output_scalar)

        scheduler.step()


        if verbose:
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.4f}')
            print(f'Train_output_scalar {train_output_scalar:.4f}')
            print(f'Training Loss:   {overall_training_loss:.4f} | Training Accuracy:   {train_total_acc:.4f}')
            print()
            print(f'Testing Loss:    {test_loss:.4f} | Testing Accuracy:   {test_acc:.4f}')
            print()

        save_name = f'model_{num_epochs}_BNN.path'
        if save:
            if save_mode == 'loss':
                if  test_loss < best_val_loss:
                    print(f"Saving new best validation loss: {overall_val_loss:.4f} < {best_val_loss:.4f}") if verbose else None


                    best_val_loss = test_loss

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


def test_Bayes(model: BNN, test_loader: DataLoader, num_mc: int, from_dnn = False, evaluate= True, verbose = True):
    '''
    model testing functionality
    '''
    with torch.no_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        calib_metric = MulticlassCalibrationError(num_classes=10, n_bins=10, norm='l1')

        label_acc = [[] for x in range(10)]
        overall_test_loss = 0
        calibrations = []
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            if hasattr(model, 'dropout'):
                if hasattr(model.dropout, "weight_matrix"):
                    with torch.no_grad():
                        model.eval()
                        first_output, _ = model(x)
                    model.dropout.weight_matrix = model.fc3.weight
                    model.dropout.prev_output = first_output

            if evaluate:
                model.eval()
            else:
                model.train()

            output_ = []
            kl_ = []

            if from_dnn:
                for mc_run in range(num_mc):
                    output = model(x, y)
                    output_.append(output)

                    kl = get_kl_loss(model)
                    kl_.append(kl)

            else:
                for mc_run in range(num_mc):

                    output, kl = model(x, y)


                    output_.append(output)

                    kl_.append(kl)
            #print(output_[0])
            test_output = torch.mean(torch.stack(output_), dim=0)

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
        overall_test_loss = overall_test_loss/len(test_loader)

        for ind, x in enumerate(label_acc):
            label_acc[ind] = sum(x)/len(x)

        total_acc = float(sum(label_acc)/10)

        calib_error = (sum(calibrations)/len(calibrations))

    if verbose:
        print(f"Testing Loss: {overall_test_loss}")
        print(f"Accuracies: {label_acc}")
        print(f"Total Accuracy: {total_acc:.4f}")
        print()

    return overall_test_loss, calib_error, total_acc, label_acc


#------- TEST the MODEL PROCESSCESS ------
# train_loader,val_loader,test_loader = loadData('CIFAR-10',batch_size= 200)

# model = BNN(in_feat= 32*32*3, out_feat=10)

# _,_,model = train_Bayes(model=model,
#                             train_loader=train_loader,
#                             test_loader=test_loader,
#                             num_epochs=50,
#                             num_mc = 10,
#                             save=True,
#                             save_mode='accuracy')


# # model.load_state_dict(torch.load("model_best_val.path"), strict=False)

# test_Bayes(model, test_loader=test_loader, num_mc=10)
