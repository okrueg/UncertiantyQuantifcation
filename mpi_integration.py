from mpi4py import MPI
from model_architectures import BasicCNN
import numpy as np
from itertools import chain

import mnist_utils

def dual_split(size, data):

    if (size/data.shape[0]) % 1 != 0:
        raise ValueError("Cannot dual split")

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

GRID_SIZE = 3
EPOCHS = 10

if rank == 0:
    data = mnist_utils.model_grid_generator((0, 0.75), (10, 512), num=grid_size)
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


local_result = mnist_utils.model_grid_training(local_chunk, EPOCHS, rank)


gathered_results = comm.gather(local_result, root=0)


if rank == 0:
    # Combine the gathered results
    combined_result = np.concatenate(gathered_results, axis=0)
    combined_result = np.reshape(combined_result, (grid_size,grid_size,2))
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
