import numpy as np
import cv2 as cv
import torch
from encoder import Encoder
from make_patches import rgb2gray
import matplotlib.pyplot as plt

test = cv.imread('test.png')
test = rgb2gray(test)
test = torch.tensor(test)
test = test[None, None, 0:160, 0:320]

encoded = np.load('encoded-patches.npy')

model = torch.load('Encoder.pth')

out, _ = model(test.float())
out = out.detach().numpy()

max_w = 0
for i in range(encoded.shape[0]):
    w = encoded[i, :] @ out[0, :]
    if w > max_w:
        max_w = w
        max_index = i

print(max_w, max_index)

patches = np.load('patches.npy')
print(patches.shape, encoded.shape)
print(patches[max_index, :, :].min())

plt.figure()
plt.imshow(patches[max_index, :, :])
plt.show()

# VIDEO_PATH = 'video.h264'

# cap = cv.VideoCapture(VIDEO_PATH)

# while (cap.isOpened()):
#     ret, frame = cap.read()

#     frame = cv.resize(frame, (600, 300))

#     cv.imshow('window', frame)

#     cv.waitKey(30)
