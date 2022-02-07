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

class Encoder(nn.Module):
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
                        nn.MaxPool2d(2),
                        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
                        nn.BatchNorm2d(256),
                        nn.MaxPool2d(2)
                    )

        self.linear1 = nn.Linear(256*5*10, 5000)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encode(x)
        B, C, W, H = x.shape
        x = torch.reshape(x, (B, C*W*H))
        x = self.relu(self.linear1(x))
        # print(f"Encoded shape: {x.shape}")
        return x, [B,C,W,H]

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, padding=2),
                        nn.BatchNorm2d(128),
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
                        nn.BatchNorm2d(1)
                    )
        self.linear2 = nn.Linear(5000, 256*5*10)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_shape):
        B,C,W,H = encoder_shape
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

def train_model(encoder, decoder, train_dataloader):
    criterion = nn.MSELoss(reduction='mean')
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.001, momentum=0.9)

    epochs = 100
    with open('best-loss.txt', 'r') as f:
        best_loss = float(f.readline())
    epoch_loss = []
    for epoch in tqdm(range(epochs)):
        train_loss = []
        for img in train_dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoded, encoded_shape = encoder(img.float())
            output = decoder(encoded, encoded_shape)

            loss = criterion(output , img.float())
            # print(loss)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            train_loss.append(loss.item())

        curr_epoch_loss = np.mean(train_loss)
        print(f"Epoch loss: {curr_epoch_loss}")
        if curr_epoch_loss < best_loss:
            torch.save(encoder, "Encoder.pth")
            best_loss = curr_epoch_loss

    return best_loss

if __name__ == '__main__':

    train_dataset = SeixalImage()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    encoder = Encoder()
    decoder = Decoder()

    best_loss = train_model(encoder, decoder, train_dataloader)

    with open('best-loss.txt', 'r+') as f:
        old_best_loss = float(f.readline())
        if best_loss < old_best_loss:
            f.truncate(0) #delete old best loss
            f.write(str(best_loss))
