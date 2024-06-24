from mpi4py import MPI

import numpy as np
from itertools import chain
import model_utils
#mpiexec -n 8 python mpi_integration.py
def dual_split(size, data):
    '''
    Can somtimes divide the data better onto the cores
    '''

    if (size/data.shape[0]) % 1 != 0:
        raise ValueError(f"Cannot dual split as {size/data.shape[0]}")

        #return np.array_split(data, size, axis=0)


    initial_chunks = np.array_split(data, data.shape[0], axis=0)
    final_chunks = []

    for chunk in initial_chunks:

        final_chunks.append(np.array_split(chunk, size/data.shape[0], axis=1))

    final_chunks = list(chain(*final_chunks))


    return final_chunks

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#if sizes are different, make sure x is the smaller one
X_SIZE = 2
Y_SIZE = 4
EPOCHS = 1
train, val, test = model_utils.loadData('CIFAR-10', batch_size=200)

if rank == 0:
    data = model_utils.model_grid_generator( x_range=(0, 1), y_range=(0, 0.75), grid_size=(X_SIZE, Y_SIZE))
    print(f'data shape: {data.shape}')

    if size > data.shape[0]:

        final_chunks = dual_split(size, data)

    else:

        final_chunks = np.array_split(data, size, axis=0)

    print("The split is")
    for i in final_chunks:
        print(i.shape)
else:
    data = None
    final_chunks = None

local_chunk = comm.scatter(final_chunks, root=0)


local_result = model_utils.model_grid_training(local_chunk,('use_reg_dropout', 'dropout_prob'), train, val, test, EPOCHS, rank)


gathered_results = comm.gather(local_result, root=0)


if rank == 0:
    # Combine the gathered results
    combined_result = np.concatenate(gathered_results, axis=0)
    combined_result = np.reshape(combined_result, (Y_SIZE, X_SIZE,2))
    print(combined_result.shape)

    val_loss_result, accuracy_result = np.array_split(combined_result, 2, axis=-1)

    val_loss_result = val_loss_result.squeeze()
    accuracy_result = accuracy_result.squeeze()


    print(f"Accuracy results:\n{accuracy_result.shape}")
    print(f"val result:\n{val_loss_result.shape}")

    np.savetxt("val_result.csv", val_loss_result, fmt='%.5f', delimiter=', ', header='Lowest Validation Loss')
    np.savetxt("acc_result.csv", accuracy_result, fmt='%.5f', delimiter=', ', header='Final Test Accuracy')
    



# model = BasicCNN(10)
# mnist_utils.train_fas_mnist(model=model,
#                             num_epochs= 2,
#                             verbose=False)

# label_acc, total = mnist_utils.test_fas_mnist(model=model,
#                                               verbose=True)

# I want to use comm scatter with a np map array
