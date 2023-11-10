from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

sobel_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])    
horizontal_edges = cv2.filter2D(gray_image, -1, sobel_kernel)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(horizontal_edges, cmap='gray')
plt.title('Horizontal Edges')
plt.axis('off')

plt.show()
