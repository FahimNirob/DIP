import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image_path = 'landscape.jpg'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_image is None:
    print("Error: Could not load the image. Please check the image path.")
else:
    last_three_bits_image = original_image & 0b111 

    difference_image = original_image - last_three_bits_image

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(last_three_bits_image, cmap='gray')
    plt.title('Last Three Bits (MSB) Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(difference_image, cmap='gray')
    plt.title('Difference Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
