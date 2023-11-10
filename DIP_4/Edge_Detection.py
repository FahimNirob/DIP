from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gradient_x = cv2.filter2D(gray_image, -1, sobel_x)
gradient_y = cv2.filter2D(gray_image, -1, sobel_y)

gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

gradient_magnitude = gradient_magnitude.astype(np.uint8)

normalized_gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

threshold = 250  
edges = (normalized_gradient_magnitude > threshold).astype(np.uint8) * 255

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(normalized_gradient_magnitude, cmap='gray')
plt.title('Normalized Gradient Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
