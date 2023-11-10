from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

histogram = np.zeros(256, dtype=int)

for pixel_value in gray_image.flatten():
    histogram[pixel_value] += 1

threshold_value = 128

thresholded_image = np.zeros_like(image, dtype=np.uint8)


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] >= threshold_value:
            thresholded_image[i, j] = 255  
        else:
            thresholded_image[i, j] = 0 


plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(histogram)
plt.title("Histogram")

plt.tight_layout()
plt.show()
