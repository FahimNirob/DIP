from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def add_salt_pepper_noise(gray_image, salt_prob, pepper_prob):
    noisy_image = gray_image.copy()
    total_pixel = gray_image.size

    number_of_salt_noise = int(total_pixel * salt_prob)
    salt_coord = [np.random.randint(0, i-1, number_of_salt_noise) for i in gray_image.shape]
    noisy_image[salt_coord[0], salt_coord[1]] = 255

    number_of_pepper_noise = int(total_pixel * pepper_prob)
    pepper_coord = [np.random.randint(0, i-1, number_of_pepper_noise) for i in gray_image.shape]
    noisy_image[pepper_coord[0], pepper_coord[1]] = 0

    return noisy_image


image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)
gray_image = cv2.resize(gray_image, (512, 512))

if gray_image is None: print("Error")

else:
    salt_prob = 0.02
    pepper_prob = 0.03
    noisy_image = add_salt_pepper_noise(gray_image, salt_prob, pepper_prob)

    filter_size = 5
    epsilon = 1e-6
    filtered_image_harmonic_mean = np.zeros_like(gray_image, dtype=np.float32)

    for i in range(filter_size // 2, noisy_image.shape[0] - filter_size // 2):
        for j in range(filter_size // 2, noisy_image.shape[1] - filter_size // 2):
            neighborhood = noisy_image[i - filter_size // 2:i + filter_size // 2 + 1, j - filter_size // 2:j + filter_size // 2 + 1]
            inverted_mean = filter_size**2 / np.sum(1.0 / (neighborhood + epsilon ))
            filtered_image_harmonic_mean[i, j] = inverted_mean

    
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Gray Image")
    plt.axis("Off")

    plt.subplot(2, 2, 3)
    plt.imshow(noisy_image, cmap='gray')
    plt.title("Noisy Image")
    plt.axis("Off")

    plt.subplot(2, 2, 4)
    plt.imshow(filtered_image_harmonic_mean, cmap='gray')
    plt.title('Harmonic Mean Filtered Image')
    plt.axis('off')

    plt.show()
