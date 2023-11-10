from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_path = './landscape.jpg'
image = Image.open(image_path)

print('Original Image Height:', image.height)
print('Original Image Width:', image.width)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image, (512, 512))

image_array = np.array(gray_image)

downscale_factors = [2, 4, 8, 16, 32]

num_rows = 2
num_cols = len(downscale_factors) // num_rows

plt.figure(figsize=(12, 6))

plt.subplot(num_rows, num_cols + 1, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Downscale the image by different factors and display side by side
for i, downscale_factor in enumerate(downscale_factors):
    downsampled_height = image_array.shape[0] // downscale_factor
    downsampled_width = image_array.shape[1] // downscale_factor

    downsampled_array = np.zeros((downsampled_height, downsampled_width), dtype=np.uint8)

    for row in range(downsampled_height):
        for col in range(downsampled_width):
            row_start = row * downscale_factor
            row_end = row_start + downscale_factor
            col_start = col * downscale_factor
            col_end = col_start + downscale_factor

            average_pixel = np.mean(image_array[row_start:row_end, col_start:col_end])

            downsampled_array[row, col] = average_pixel.astype(np.uint8)

    plt.subplot(num_rows, num_cols + 1, i + 2)
    plt.imshow(downsampled_array, cmap='gray')
    plt.title(f'Downscale {downscale_factor}x')
    plt.axis('off')

plt.tight_layout()
plt.show()
