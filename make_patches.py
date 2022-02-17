import rasterio
import numpy as np
import matplotlib.pyplot as plt

def read_image():
    with rasterio.open('seixal_modified.tif', 'r') as ds:
        img = ds.read()  # read all raster values
    return np.moveaxis(img, 0, 2)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def crop_image(img):
    crop_x, crop_y = 40, 80
    h, w = img.shape

    n_h, n_w = h//crop_x, w//crop_y
    all_patches = []
    for i in range(0, h, crop_x):
        for j in range(0, w, crop_y):
            patch = img[i:i+crop_x, j:j+crop_y]
            if (patch.shape[0] != crop_x) or (patch.shape[1] != crop_y):
                continue
            all_patches.append(patch)

    return np.array(all_patches)

if __name__ == '__main__':
    img = read_image()
    img = rgb2gray(img)

    patches = crop_image(img)

    np.save('patches-small', patches)

    print(patches.shape)
