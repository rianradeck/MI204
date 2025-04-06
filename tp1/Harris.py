import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
# Mettre ici le calcul de la fonction d'intérêt de Harris

# Calcul des gradients Ix et Iy
Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Calcul des termes de la matrice M (produits des gradients)
Ix2 = Ix**2
Iy2 = Iy**2
Ixy = Ix*Iy

# Paramètres pour le calcul de Harris
k = 0.04  # Coefficient empirique généralement entre 0.04 et 0.06
window_size = 3  # Taille de la fenêtre W

# Application d'un filtre gaussien pour le lissage (équivalent à la somme pondérée dans la fenêtre W)
Ix2 = cv2.GaussianBlur(Ix2, (window_size, window_size), 1)
Iy2 = cv2.GaussianBlur(Iy2, (window_size, window_size), 1)
Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 1)

# Calcul de la fonction d'intérêt de Harris (Theta)
det_M = Ix2 * Iy2 - Ixy**2
trace_M = Ix2 + Iy2
Theta = det_M - k * trace_M**2

# Normalisation de Theta entre 0 et 1 pour faciliter le seuillage
Theta = (Theta - Theta.min()) / (Theta.max() - Theta.min())

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.3
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

print("Valeurs min/max de Theta:", Theta.min(), Theta.max())
print("Nombre de points détectés:", np.count_nonzero(Theta_maxloc))

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Relecture image pour affichage couleur
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

plt.show()
