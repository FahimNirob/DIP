from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


def manual_erosion(binary_image, kernel):
    eroded_image = binary_image.copy()
    height, width = binary_image.shape

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            if binary_image[i,j] == 255:
                region = binary_image[i: i + kernel_size, j: j + kernel_size]
                if np.all(np.logical_and(region, kernel)):
                    eroded_image[i, j] = 255
    
    return eroded_image

def manual_dilation(binary_image, kernel):
    dilated_image = binary_image.copy()
    height, width = binary_image.shape

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            if binary_image[i,j] == 255:
                region = binary_image[i: i + kernel_size, j: j + kernel_size]
                dilated_image[i: i + kernel_size, j: j + kernel_size] = np.logical_or(dilated_image[i: i + kernel_size, j: j + kernel_size], kernel)
    
    dilated_image = np.where(dilated_image == True, 255, 0)
    return dilated_image

def manual_open(binary_image, kernel):
    eroded = manual_erosion(binary_image, kernel)
    opened_image = manual_dilation(eroded, kernel)
    return opened_image

def manual_close(binary_image, kernel):
    dilated = manual_dilation(binary_image, kernel)
    closed_image = manual_erosion(dilated, kernel)
    return closed_image

image_path = 'A.png'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

_, binary_image = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

eroded_image = manual_erosion(binary_image, kernel)
dilated_image = manual_dilation(binary_image, kernel)
opened_image = manual_open(binary_image, kernel)
closed_image = manual_close(binary_image, kernel)

plt.figure(figsize=(12, 6))

plt.subplot(2,2,1)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')
plt.axis('Off')

plt.subplot(2,2,2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('Off')

plt.subplot(2,2,3)
plt.imshow(opened_image, cmap='gray')
plt.title('Open Image')
plt.axis('Off')

plt.subplot(2,2,4)
plt.imshow(closed_image, cmap='gray')
plt.title('Close Image')
plt.axis('Off')

plt.show()
