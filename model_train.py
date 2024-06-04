import torch
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

from model_architectures import basic_nn, dimension_increaser
from data_set_generation import generate_data, plot_data_static

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def train(x: np.ndarray,
          y: np.ndarray,
          shift: np.ndarray,
          model: basic_nn,
          dim_inc: dimension_increaser,
          num_epochs = 40,
          batch_size = 10,
          hidden_dim = 20,
          in_dims = 10,
          lr = 0.1):

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    shift = torch.from_numpy(shift).float()

    x_class1 = x[y == 1]

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = torch.nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    #scheduler = ExponentialLR(optimizer,0.95)


    #-----Train ----
    model.train()
    for epoch in range(num_epochs):
        overall_loss = 0
        total_embeddings = None
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            #x = dim_inc(x)
            output = model(x)
            output = output.view(-1)

            loss = loss_fn(output, y)
            
            overall_loss += loss.item()
            
            loss.backward()

            optimizer.step()

        # get embeddings for dropout
        shift_dist = model.forwardEmbeddings(shift)
        total_embeddings = model.forwardEmbeddings(x_class1)

        # update dropout distributions
        model.dropout.update_distribution(total_embeddings, train_dist=True)
        model.dropout.update_distribution(shift_dist, train_dist=False)

        # decay alpha value to slowly introduce dropout
        model.dropout.alpha = model.dropout.alpha * .75
        #scheduler.step()
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    #----Test ----
def test(x: np.ndarray, model: basic_nn, dim_inc: dimension_increaser):
    x = torch.from_numpy(x).float()
    model.eval()
    with torch.no_grad():
        #x = dim_inc(x)
        x = model(x)
    return x

def feature_correlation(x: np.ndarray, model: basic_nn, dim_inc: dimension_increaser):
    x = torch.from_numpy(x).float()
    #x = dim_inc(x)
    x = model.forwardEmbeddings(x).detach().numpy()
    cor = np.corrcoef(x, rowvar=False)
    return cor

def compress_features(x: np.ndarray, model: basic_nn, dim_inc: dimension_increaser, type = 'lda'):
    x = torch.from_numpy(x).float()
    #x = dim_inc(x)
    y = model.forward(x).detach()
    y = y.view(-1).numpy()
    y = np.round(y)
    x = model.forwardEmbeddings(x).detach().numpy()

    result = None
    if type == 'pca':
        pca = PCA(n_components=2)
        result = pca.fit_transform(x)
    elif type == 'lda':
        lda = LinearDiscriminantAnalysis(n_components=1)
        result = lda.fit_transform(x,y)
    return result

