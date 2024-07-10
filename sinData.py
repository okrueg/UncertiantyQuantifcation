import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Step 1: Generate the data
x_obs = np.hstack([np.linspace(-0.2, 0.2, 1000)])#, np.linspace(0.6, 1, 500)])
noise = 0.02 * np.random.randn(x_obs.shape[0])
y_obs = np.sin(2 * np.pi * (x_obs + noise))

x_true = np.linspace(-0.5, 1.5, 1000)
y_true =np.sin(2 * np.pi * x_true)

# Step 2: Convert to PyTorch tensors
x_obs_tensor = torch.tensor(x_obs, dtype=torch.float32).view(-1, 1)
y_obs_tensor = torch.tensor(y_obs, dtype=torch.float32).view(-1, 1)

x_true_tensor = torch.tensor(x_true, dtype=torch.float32).view(-1, 1)
y_true_tensor = torch.tensor(y_true, dtype=torch.float32).view(-1, 1)


# print(x_obs_tensor.shape)
# exit()
# Step 3: Define the custom dataset
class SineWaveDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Step 4: Create train and test DataLoaders
# Split the data
train_size = int(0.8 * len(x_obs_tensor))
test_size = len(x_obs_tensor) - train_size

train_indices = list(range(train_size))
test_indices = list(range(train_size, len(x_obs_tensor)))

x_train_tensor = torch.index_select(x_obs_tensor, 0, torch.tensor(train_indices))
y_train_tensor = torch.index_select(y_obs_tensor, 0, torch.tensor(train_indices))

x_test_tensor = torch.index_select(x_obs_tensor, 0, torch.tensor(test_indices))
y_test_tensor = torch.index_select(y_obs_tensor, 0, torch.tensor(test_indices))

# Create datasets
train_dataset = SineWaveDataset(x_train_tensor, y_train_tensor)
test_dataset = SineWaveDataset(x_test_tensor, y_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 5: Visualize the data (Optional)
# plt.figure(figsize=(10, 6))
# plt.scatter(x_obs, y_obs, label='Noisy Observations', color='blue', s=10)
# plt.plot(x_true, y_true, label='True Sine Wave', color='red')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Noisy Observations and True Sine Wave')
# plt.show()
