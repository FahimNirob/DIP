from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

last_three_bits_image = gray_image & 0b111
differenced_image = gray_image - last_three_bits_image


plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')
plt.axis('Off')

plt.subplot(1,3,2)
plt.imshow(last_three_bits_image, cmap='gray')
plt.title('Gamma')
plt.axis('Off')

plt.subplot(1,3,3)
plt.imshow(differenced_image, cmap='gray')
plt.title('Inverse Log')
plt.axis('Off')


plt.show()