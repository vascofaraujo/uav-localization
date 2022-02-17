import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from encoder import Encoder, SeixalImage

if __name__=='__main__':

    train_dataset = SeixalImage()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    encoder = torch.load('Encoder.pth')

    data = np.load('patches.npy')
    N,W,H = data.shape
    encoded_patches = np.zeros((N,5000))

    encoder.eval()
    for i, img in enumerate(train_dataloader):
        with torch.no_grad():
            encoded, _ = encoder(img.float())


            encoded_patches[i,:] = encoded[0,:]
    np.save('encoded-patches.npy', encoded_patches)
    print("Bye bye")
