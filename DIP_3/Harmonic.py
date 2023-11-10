from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def salt_pepper_image(gray_image, salt_prob, pepper_prob):
    noisy_image = gray_image.copy()
    total_pixel = gray_image.size

    num_of_salt_noise = int(total_pixel * salt_prob)
    salt_coord = [np.random.randint(0, i-1, num_of_salt_noise) for i in gray_image.shape]
    noisy_image[salt_coord[0], salt_coord[1]] = 255

    num_of_pepper_noise = int(total_pixel * pepper_prob)
    pepper_coord = [np.random.randint(0, i-1, num_of_pepper_noise) for i in gray_image.shape]
    noisy_image[pepper_coord[0], pepper_coord[1]] = 0

image_path = 'landscape.jpg'
rgb_image = Image.open(image_path)

gray_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)
gray_image = cv2.resize(gray_image, (512, 512))


salt_prob = 0.02
pepper_prob = 0.04
noisy_image = salt_pepper_image(gray_image, salt_prob, pepper_prob)

