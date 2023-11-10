from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

gamma = .1

power_law_transformed = np.power(gray_image / float(np.max(gray_image)), gamma)
power_law_transformed = np.uint8(power_law_transformed * 255)

original_image_float = gray_image.astype(np.float32)

inverse_log_transformed = cv2.log(1 + original_image_float)
inverse_log_transformed = np.uint8((inverse_log_transformed / np.max(inverse_log_transformed)) * 255)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(power_law_transformed, cmap='gray')
plt.title('Power-Law Transformed')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(inverse_log_transformed, cmap='gray')
plt.title('Inverse Logarithmic Transformed')
plt.axis('off')

plt.tight_layout()
plt.show()
