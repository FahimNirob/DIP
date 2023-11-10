from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_path ='landscape.jpg'
rgb_image = Image.open(image_path)

image_array = np.array(rgb_image)
image_shape = image_array.shape
print("RGB Image Shape: ", image_shape)

grayscale = np.zeros_like(image_array[:, :, 0])

for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        red = image_array[i, j, 0]
        green = image_array[i, j, 1]
        blue = image_array[i, j, 2]
        
        gray_value = int(0.29*red + 0.58*green + 0.11*blue)
        grayscale[i,j] = gray_value

plt.subplot(2,2,1)
plt.imshow(rgb_image)
plt.title('RGB')

plt.subplot(2,2,2)
plt.imshow(grayscale, cmap='gray')
plt.title('Grayscale')

plt.show()