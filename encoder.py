import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        patches = (patches - np.mean(patches)) / (np.std(patches))
        patches =  torch.from_numpy(patches[patch_index, :, :])
        return patches[None, :, :]

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
                        nn.BatchNorm2d(16),
                        nn.MaxPool2d(2),
                        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
                        nn.BatchNorm2d(32),
                        nn.MaxPool2d(2),
                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2),
                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2)
                    )
        #4, 512, 6, 16
        self.decode = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, padding=2),
                        nn.BatchNorm2d(64),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
                        nn.BatchNorm2d(32),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
                        nn.BatchNorm2d(16),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5, padding=2),
                        nn.BatchNorm2d(1),
                    )
        self.linear1 = nn.Linear(128*10*20, 5000)
        self.linear2 = nn.Linear(5000, 128*10*20)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.encode(x)
        B, C, W, H = x.shape
        x = torch.reshape(x, (B, C*W*H))
        x = self.relu(self.linear1(x))
        # print(f"Encoded shape: {x.shape}")
        x = self.relu(self.linear2(x))
        x = torch.reshape(x, (B, C, W, H))
        x = self.decode(x)
        return x

def show_images(output, img):
    fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1, 2, 1)
    plt.imshow(output[0, 0, :, :].detach().numpy())
    fig.add_subplot(1,2,2)
    plt.imshow(img[0,0,:,:])
    plt.show()

def train_model(model, train_dataloader):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 20
    best_loss = 0.2
    epoch_loss = []
    for epoch in tqdm(range(epochs)):
        train_loss = []
        for img in train_dataloader:
            optimizer.zero_grad()
            output = model(img.float())

            loss = criterion(output , img.float())
            # print(loss)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        curr_epoch_loss = np.mean(train_loss)
        print(f"Epoch loss: {curr_epoch_loss}")
        if curr_epoch_loss < best_loss:
            torch.save(model, "AutoEncoder.pth")
            best_loss = curr_epoch_loss

if __name__ == '__main__':

    train_dataset = SeixalImage()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = AutoEncoder()

    _ = train_model(model, train_dataloader)
