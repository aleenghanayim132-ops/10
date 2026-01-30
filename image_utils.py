from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    return np.array(Image.open(path).convert('RGB'))

def edge_detection(image_array):
    weights = np.array([0.2989, 0.5870, 0.1140])
    image_gray = np.dot(image_array[..., :3], weights)

    filterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filterY = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]])

    gx = convolve2d(image_gray, filterX, mode='same', boundary='symm')
    gy = convolve2d(image_gray, filterY, mode='same', boundary='symm') 
  
    return np.sqrt(gx**2 + gy**2)
