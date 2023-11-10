from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

threshold_value = 200

thresholded_image = np.where(gray_image > threshold_value, 255, 0).astype(np.uint8)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.show()
