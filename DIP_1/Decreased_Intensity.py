from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

image_path = './landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image, (512, 512))

image_array = np.array(gray_image)

num_bits = int(np.log2(np.max(image_array)) + 1)
print(num_bits)

plt.figure(figsize=(15, 10))

for i in range(num_bits):
    shifted_image = (image_array >> i).astype(np.uint8)
    title = f'Bit {i}'
  
    plt.subplot(1, num_bits, i + 1)
    plt.imshow(shifted_image, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()


