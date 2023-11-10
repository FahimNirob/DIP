from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

histogram = np.zeros(256, dtype=np.uint8)

for pixel_value in gray_image.flatten():
    histogram[pixel_value] += 1

threshold = 100

threshold_image = gray_image.copy()

for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        if (gray_image[i,j]) >= threshold:
            threshold_image[i,j] = 255
        else:
            threshold_image[i,j] = 0

threshold_histogram = np.zeros(256, dtype=np.uint8)

for pixel_value in threshold_image.flatten():
    threshold_histogram[pixel_value] += 1


plt.figure(figsize=(12, 6))

plt.subplot(2,2,1)
plt.imshow(gray_image, cmap='gray')
plt.title("Gray Image")

plt.subplot(2,2,2)
plt.bar(range(256), histogram, color='g', alpha=0.9)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)

plt.subplot(2,2,3)
plt.imshow(threshold_image, cmap='gray')
plt.title("Threshold image")

plt.subplot(2,2,4)
plt.bar(range(256), threshold_histogram, color='g', alpha=0.9)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)


plt.show()





