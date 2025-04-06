import numpy as np
import cv2
from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img = np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png', 0))
(h, w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

# Noyaux de Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Calcul des gradients
I_x = cv2.filter2D(img, -1, sobel_x)
I_y = cv2.filter2D(img, -1, sobel_y)
gradient_magnitude = np.sqrt(I_x**2 + I_y**2)

# Normalisation spécifique pour les gradients (conserve le signe)
def normalize_gradient(grad):
    abs_max = np.max(np.abs(grad))
    return (grad + abs_max) / (2 * abs_max)  # Ramène entre 0 et 1

# Normalisation pour la norme
def normalize_magnitude(mag):
    return (mag - np.min(mag)) / (np.max(mag) - np.min(mag))

# Affichage OpenCV
cv2.imshow('I_x (Gris)', (255*normalize_gradient(I_x)).astype(np.uint8))
cv2.imshow('I_y (Gris)', (255*normalize_gradient(I_y)).astype(np.uint8))
cv2.imshow('Norme', (255*normalize_magnitude(gradient_magnitude)).astype(np.uint8))
cv2.waitKey(0)

# Affichage Matplotlib
plt.figure(figsize=(15,5))

plt.subplot(131)
plt.imshow(normalize_gradient(I_x), cmap='gray', vmin=0, vmax=1)
plt.title('Dérivée horizontale (I_x)')

plt.subplot(132)
plt.imshow(normalize_gradient(I_y), cmap='gray', vmin=0, vmax=1)
plt.title('Dérivée verticale (I_y)')

plt.subplot(133)
plt.imshow(normalize_magnitude(gradient_magnitude), cmap='gray', vmin=0, vmax=1)
plt.title('Norme du gradient')

plt.tight_layout()
plt.show()