import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WindDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt("./data/wine/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
dataset = WindDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

datatiter = iter(dataloader)
data = datatiter.next()
features, labels = data
print(features, labels)