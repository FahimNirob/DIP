from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    total_pixels = image.size

    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255



    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

if gray_image is None:
    print("Error:")
else:
    salt_prob = 0.05  
    pepper_prob = 0.05 
    noisy_image = add_salt_and_pepper_noise(gray_image, salt_prob, pepper_prob)

   
    average_kernel = np.ones((5, 5), dtype=np.float32) / 25
    average_filtered = cv2.filter2D(noisy_image, -1, average_kernel)

    median_filtered = cv2.medianBlur(noisy_image, 5)

    average_mse = np.mean((gray_image.astype(float) - average_filtered.astype(float))**2)
    average_PSNR = 20 * np.log(255/np.sqrt(average_mse))

    median_mse = np.mean((gray_image.astype(float) - median_filtered.astype(float))**2)
    median_PSNR = 20 * np.log(255/np.sqrt(median_mse))

  
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Gray Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.imshow(average_filtered, cmap='gray')
    plt.title(f'Average Filter (PSNR: {average_PSNR: .2f} dB)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(median_filtered, cmap='gray')
    plt.title(f'Median Filter (PSNR: {median_PSNR: .2f} dB)')
    plt.axis('off')


    plt.tight_layout()
    plt.show()
