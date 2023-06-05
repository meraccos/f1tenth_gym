import cv2
import matplotlib.pyplot as plt
import numpy as np

for idx in range(50):
    img = np.array(plt.imread(f'maps_handmade/map{idx}.png'))
    img_gray = img.sum(2) / 3
    # mask = img_gray > 100
    # img_ext = np.zeros_like(img_gray)
    # img_ext[np.where(img_gray > 50)] = 255
    # img_ext[mask] = 255
    img_ext = img_gray
    print(img_gray.max())
    plt.imsave(f"maps_handmade_gray/map{idx}.png", img_ext.astype(np.uint8), cmap='gray')