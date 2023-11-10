from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor((np.array(image)), cv2.COLOR_RGB2GRAY)

min = 50
max = 150

enhancement_factor = 2

enhanced_image = gray_image.copy() 
enhanced_image[(gray_image >= min) & (gray_image <= max)] *= enhancement_factor

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')
plt.axis('off')

plt.show()
