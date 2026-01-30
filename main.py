#note; I used Ai for a clearer and better looking project
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

# 1. Load the original color image 
image_path = 'CAT.jpg' 
original_image = load_image(image_path)

# 2. Suppress noise using a median filter
# This replaces pixels with the neighborhood median to remove random noise [6].
clean_image = median(original_image, ball(3))

# 3. Detect edges on the noise-free image
# This calculates the gradient intensity for every pixel [2].
edgeMAG = edge_detection(clean_image)

# 4. Thresholding to create a binary array
# Use a histogram to find a threshold that separates edges from background noise [9].
plt.figure()
plt.hist(edgeMAG.flatten(), bins=100, log=True)
plt.title("Edge Intensity Histogram")
plt.show()

# Set a threshold value based on your histogram observation
threshold_value = 50  # Example value: adjust this based on your image
edge_binary = edgeMAG > threshold_value

# 5. Display and save the result
plt.imshow(edge_binary, cmap='gray')
plt.title("Binary Edge Detection")
plt.axis('off')
plt.show()

# Convert boolean/binary array to an image and save it as .png [7]
edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save('my_edges.png')
