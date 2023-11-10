from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def add_salt_pepper_noise(gray_image, salt_prob, pepper_prob):
    noisy_image = gray_image.copy()
    total_pixel = gray_image.size

    number_of_salt_noise = int(total_pixel * salt_prob)
    salt_coord = [np.random.randint(0, i-1, number_of_salt_noise) for i in gray_image.shape]
    noisy_image [salt_coord[0], salt_coord[1]] = 255

    number_of_pepper_noise = int(total_pixel * pepper_prob)
    pepper_coord = [np.random.randint(0, i-1, number_of_pepper_noise) for i in gray_image.shape]
    noisy_image [pepper_coord[0], pepper_coord[1]] = 0

    return noisy_image
 

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)
gray_image = cv2.resize(gray_image, (512, 512))


if gray_image is None:
    print("Error")
else:
    salt_prob = 0.02
    pepper_prob = 0.03
    noisy_image = add_salt_pepper_noise(gray_image, salt_prob, pepper_prob)

    #Mask 3x3
    kernel_1 = np.ones((3,3), dtype=np.float32) / 9
    filtered_image_1 = cv2.filter2D(noisy_image, -1, kernel_1)
    avg_MSE_1 = np.mean((gray_image.astype(float) - noisy_image.astype(float))**2)
    avg_PSNR_1 = 20 * np.log(255/np.sqrt(avg_MSE_1))
    
    #Mask 5x5
    kernel_2 = np.ones((5,5), dtype=np.float32) / 25
    filtered_image_2 = cv2.filter2D(noisy_image, -1, kernel_2)
    avg_MSE_2 = np.mean((gray_image.astype(float) - noisy_image.astype(float))**2)
    avg_PSNR_2 = 20 * np.log(255 / np.sqrt(avg_MSE_2))

    #Mask 7x7
    kernel_3 = np.ones((7,7), dtype=np.float32) / 49
    filtered_image_3 = cv2.filter2D(noisy_image, -1, kernel_3)
    avg_MSE_3 = np.mean((gray_image.astype(float) - noisy_image.astype(float))**2)
    avg_PSNR_3 = 20 * np.log(255 / np.sqrt(avg_MSE_3))


    plt.subplot(2,3,1)
    plt.imshow(gray_image, cmap='gray')
    plt.title("Gray Image")
    plt.axis('Off')

    plt.subplot(2,3,2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title("Noisy Image")
    plt.axis("Off")

    plt.subplot(2,3,4)
    plt.imshow(filtered_image_1, cmap='gray')
    plt.title(f'Mask 3x3(PSNR_1: {avg_PSNR_1: .2f} dB)')
    plt.axis("Off")

    plt.subplot(2,3,5)
    plt.imshow(filtered_image_2, cmap='gray')
    plt.title(f'Mask 5x5(PSNR_2: {avg_PSNR_2: .2f} dB)')
    plt.axis("Off")

    plt.subplot(2,3,6)
    plt.imshow(filtered_image_3, cmap='gray')
    plt.title(f'Mask 7x7(PSNR_3: {avg_PSNR_3: .2f} dB)')
    plt.axis("Off")

    plt.show()