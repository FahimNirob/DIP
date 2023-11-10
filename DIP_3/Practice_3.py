from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def add_salt_pepper_noise(gray_image, salt_prob, pepper_prob):
    noisy_image = gray_image.copy()
    total_pixel = gray_image.size

    num_salt = int(total_pixel * salt_prob)
    salt_coord = [np.random.randint(0, i-1, num_salt) for i in gray_image.shape]
    noisy_image [salt_coord[0], salt_coord[1]] = 255

    num_pepper = int(total_pixel * pepper_prob)
    pepper_coord = [np.random.randint(0, i-1, num_pepper) for i in gray_image.shape]
    noisy_image [pepper_coord[0], pepper_coord[1]]

    return noisy_image

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)


if gray_image is None:
    print("Error")
else:
    salt_prob = 0.2
    pepper_prob = 0.3 

    noisy_image = add_salt_pepper_noise(gray_image, salt_prob, pepper_prob)

    average_kernel = np.ones((5,5), dtype=np.uint32) / 25
    average_filtered = cv2.filter2D(gray_image, -1, average_kernel)

    median_filtered = cv2.medianBlur(noisy_image, 5)

    plt.figure(figsize=(12,6))

    plt.subplot(2,2,1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Gray Image")
    plt.axis('Off')

    plt.subplot(2,2,2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title("Noisy Image")
    plt.axis('Off')

    plt.subplot(2,2,3)
    plt.imshow(average_filtered, cmap='gray')
    plt.title("Average Filtered")
    plt.axis('Off')

    plt.subplot(2,2,4)
    plt.imshow(median_filtered, cmap='gray')
    plt.title("Median")
    plt.axis('Off')

    plt.show()


