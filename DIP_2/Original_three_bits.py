from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

original_image_path = 'landscape.jpg'
original_image = cv2.imread(original_image_path)

last_3_bits_image = np.zeros_like(original_image)

difference_image = np.zeros_like(original_image)

for i in range(original_image.shape[0]):
    for j in range(original_image.shape[1]):
        for channel in range(original_image.shape[2]):
            pixel_value = original_image[i, j, channel]
            
            # Extract the last 3 bits
            last_3_bits = pixel_value & 0b00000111
            
            # Shift the last 3 bits back to their original position
            last_3_bits <<= 5
            
            # Assign the last 3 bits to the corresponding pixel in the last 3 bits image
            last_3_bits_image[i, j, channel] = last_3_bits
            
            # Calculate the absolute difference
            difference = abs(pixel_value - last_3_bits)
            
            # Assign the difference to the corresponding pixel in the difference image
            difference_image[i, j, channel] = difference

# Display the original, last 3 bits, and difference images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(last_3_bits_image, cv2.COLOR_BGR2RGB))
plt.title('Last 3 Bits Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(difference_image, cv2.COLOR_BGR2RGB))
plt.title('Difference Image')
plt.axis('off')

plt.tight_layout()
plt.show()


# 2TB - 400GB, 100GB, 200GB, 1.1TB
# 2TB 