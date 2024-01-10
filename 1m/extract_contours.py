import cv2
import os
import numpy as np
import shutil
src_dir = 'shapes'

dst_dir = 'contour_shapes'

if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)

os.makedirs(dst_dir)

for folder in os.listdir(src_dir):
    if not os.path.exists(os.path.join(dst_dir, folder)):
        os.makedirs(os.path.join(dst_dir, folder))

    for filename in os.listdir(os.path.join(src_dir, folder)):
        image = cv2.imread(os.path.join(src_dir, folder, filename), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255), 3)  # Увеличение толщины контура до 3 пикселей
        contour_image = cv2.bitwise_not(contour_image)  # Инвертирование цветов изображения
        resized_image = cv2.resize(contour_image, (28, 28))
        cv2.imwrite(os.path.join(dst_dir, folder, filename), contour_image)

        # Вывод размера изображения
        print(f"Saved image {filename} with size {resized_image.shape}")