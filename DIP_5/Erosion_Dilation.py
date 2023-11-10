from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_erosion(image, kernel):
    eroded_image = np.zeros_like(image)
    height, width = image.shape

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            if image[i, j] == 255:
                region = image[i:i + kernel_size, j:j + kernel_size]
                if np.all(np.logical_and(region, kernel)):
                    eroded_image[i, j] = 255

    return eroded_image


def manual_dilation(image, kernel):
    dilated_image = np.zeros_like(image)
    height, width = image.shape

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            if image[i, j] == 255:
                region = image[i:i + kernel_size, j:j + kernel_size]
                dilated_image[i:i + kernel_size, j:j + kernel_size] = np.logical_or(dilated_image[i:i + kernel_size, j:j + kernel_size], kernel)

    dilated_image = np.where(dilated_image == True, 255, 0)
    return dilated_image

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

_, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

kernel_size = 3  
kernel = np.ones((kernel_size, kernel_size), np.uint8)

eroded_image = manual_erosion(binary_image, kernel)
dilated_image = manual_dilation(binary_image, kernel)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(rgb_image)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(binary_image, cmap='gray', vmin=0, vmax=255)
plt.title('Binary Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(eroded_image, cmap='gray', vmin=0, vmax=255)
plt.title('Manually Eroded Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(dilated_image, cmap='gray', vmin=0, vmax=255)
plt.title('Manually Dilated Image')
plt.axis('off')

plt.tight_layout()
plt.show()
