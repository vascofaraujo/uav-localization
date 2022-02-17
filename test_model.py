import numpy as np
import cv2 as cv
import torch
from encoder import Encoder, SeixalImage
from make_patches import rgb2gray

test = cv.imread('test.png')
test = rgb2gray(test)
test = torch.tensor(test)
test = test[None, None, 0:160, 0:320]

encoded = np.load('encoded-patches.npy')

model = torch.load('Encoder.pth')

out, out_shape = model(test.float())
print(out_shape)
print(test.shape, encoded.shape)
print(model)
