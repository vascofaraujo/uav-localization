import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SeixalImage(Dataset):
    def __init__(self):
        self.crop_x = 160
        self.crop_y = 320
        self.height = 1796
        self.width = 4973

    def __len__(self):
        return (self.height//self.crop_x)*(self.width//self.crop_y)

    def __getitem__(self, patch_index):
        patches = np.load('patches.npy')
        return torch.from_numpy(patches[patch_index, :, :])

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


if __name__ == '__main__':

    train_dataset = SeixalImage()
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)

    for i in train_dataloader:
        print(i.shape)
