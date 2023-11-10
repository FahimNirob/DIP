from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

image_array = np.array(rgb_image)
image_shape = image_array.shape

gray_image = np.zeros_like(image_array[:, :, 0])

for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        red = image_array[i, j, 0]
        green = image_array[i, j, 1]
        blue = image_array[i, j, 2]

        gray_value = int(red * .29 + green * .58 + blue * .11)
        gray_image[i,j] = gray_value

plt.subplot(1,2,1)
plt.imshow(rgb_image)

plt.subplot(1,2,2)
plt.imshow(gray_image, cmap='gray')

plt.show()
