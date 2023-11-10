from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


image_path = './landscape.jpg'
image = Image.open(image_path)

print('Original Image Height: ',image.height)
print('Original Image Width: ',image.width)

downscale_factor = 8

# Convert the image to a NumPy array
image_array = np.array(image)

downsampled_height = image_array.shape[0] // downscale_factor
downsampled_width = image_array.shape[1] // downscale_factor

print('Downsampled Height: ',downsampled_height)
print('Downsampled Width: ',downsampled_width)

downsampled_array = np.zeros((downsampled_height, downsampled_width, 3), dtype=np.uint8)

for i in range (downsampled_height):
    for j in range (downsampled_width):
        row_start = i * downscale_factor
        row_end = row_start + downscale_factor
        col_start = j * downscale_factor
        col_end = col_start + downscale_factor

        average_pixel = np.mean(image_array[row_start:row_end, col_start:col_end], axis=(0,1))

        downsampled_array[i, j] = average_pixel.astype(np.uint8)

# Convert the NumPy array back to an image
downsampled_image = Image.fromarray(downsampled_array)



plt.subplot(1,2,1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(downsampled_image)
plt.title('Down Sampled Image')

plt.show()
