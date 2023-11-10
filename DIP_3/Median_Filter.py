import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    total_pixels = image.size

    num_salt = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255

    num_pepper = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_image

image_path = 'landscape.jpg'
original_image = cv2.imread(image_path)

if original_image is None:
    print("Error: Could not load the image. Please check the image path.")
else:
    salt_prob = 0.02  
    pepper_prob = 0.02  
    noisy_image = add_salt_and_pepper_noise(original_image, salt_prob, pepper_prob)

   
    average_kernel = np.ones((5, 5), dtype=np.float32) / 25
    average_filtered = cv2.filter2D(noisy_image, -1, average_kernel)

   
    median_filtered = cv2.medianBlur(noisy_image, 5)

    
    mse_average = np.mean((original_image.astype(float) - average_filtered.astype(float))**2)
    psnr_average = 20 * np.log10(255 / np.sqrt(mse_average))

  
    mse_median = np.mean((original_image.astype(float) - median_filtered.astype(float))**2)
    psnr_median = 20 * np.log10(255 / np.sqrt(mse_median))

  
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    plt.title('Noisy Image')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(average_filtered, cv2.COLOR_BGR2RGB))
    plt.title(f'Average Filter (PSNR: {psnr_average:.2f} dB)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))
    plt.title(f'Median Filter (PSNR: {psnr_median:.2f} dB)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
