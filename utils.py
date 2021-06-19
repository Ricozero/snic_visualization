import numpy as np
from skimage.segmentation import mark_boundaries
from skimage import io
import matplotlib.pyplot as plt

def show_bounaries(img, labels, save=None):
    bounds = mark_boundaries(img, labels, mode="inner")
    plt.imshow(bounds)
    plt.show()
    if save:
        io.imsave(save, (bounds*255).astype(np.uint8))

def save_boundaries(img, labels, filename):
    bounds = mark_boundaries(img, labels, mode="inner")
    io.imsave(filename, (bounds*255).astype(np.uint8))