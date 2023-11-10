from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

image_array = np.array(rgb_image)
rgb_shape = image_array.shape
print(rgb_shape)

width, height = rgb_image.size
split_point = width // 2

left_part = rgb_image.crop((0, 0, split_point, height))
right_part = rgb_image.crop((split_point, 0, width, height))


plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.imshow(rgb_image)

plt.subplot(2,2,3)
plt.imshow(left_part)

plt.subplot(2,2,4)
plt.imshow(right_part)

plt.show()