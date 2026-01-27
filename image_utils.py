from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    image = Image.open(path)
    image_array = np.array(image)
    return image_array

def edge_detection(image):
    image_array = Image.open(image).convert('L')
    image_gray = np.array(image_array)
    filterX = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    filterY = np.array([[1,2,1],
                        [0,0,0],
                        [-1,-2,-1]])
    filtered_imageX = convolve2d(image_gray,filterX, mode='same' , boundary='symm')
    filtered_imageY = convolve2d(image_gray,filterY, mode='same' , boundary='symm') 
    filtered_image = np.sqrt(filtered_imageX**2 + filtered_imageY**2)
    return filtered_image
