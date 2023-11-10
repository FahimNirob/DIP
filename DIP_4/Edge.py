from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

image_path = 'landscape.jpg'
image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

sobel_h = np.array([[-1,-2,-1], [0, 0, 0], [1, 2, 1]])
sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

horizontal_edges = cv2.filter2D(gray_image, -1, sobel_h)
vertical_edges = cv2.filter2D(gray_image, -1, sobel_v)

edges = horizontal_edges + vertical_edges
sharp_image = gray_image + edges

plt.figure(figsize = (12,6))

plt.subplot(2,3,1)
plt.imshow(gray_image, cmap='gray')
plt.title("Original Image")
plt.axis("Off")

plt.subplot(2,3,3)
plt.imshow(vertical_edges, cmap='gray')
plt.title("Vertical Edges")
plt.axis("Off")

plt.subplot(2,3,4)
plt.imshow(horizontal_edges, cmap='gray')
plt.title("Horizontal Edges")
plt.axis("Off")

plt.subplot(2,3,5)
plt.imshow(edges, cmap='gray')
plt.title("Vertical + Horizontal")
plt.axis("Off")

plt.subplot(2,3,2)
plt.imshow(sharp_image, cmap='gray')
plt.title("Sharpened Image")
plt.axis("Off")



plt.show()