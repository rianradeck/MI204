import numpy as np
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)

#Lecture de la paire d'images
img1 = cv2.imread('../Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
img2 = cv2.imread('../Image_Pairs/torb_small2.png')
print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)

#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 250,#Par défaut : 500
                       scaleFactor = 2,#Par défaut : 1.2
                       nlevels = 3)#Par défaut : 8
  kp2 = cv2.ORB_create(nfeatures=250,
                        scaleFactor = 2,
                        nlevels = 3)
  print("Détecteur : ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false
    		        threshold = 0.001,#Par défaut : 0.001
  		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection des keypoints
pts1 = kp1.detect(gray1,None)
pts2 = kp2.detect(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection des points d'intérêt :",time,"s")

#Affichage des keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags définit le niveau d'information sur les points d'intérêt
# 0 : position seule ; 4 : position + échelle + direction
img2 = cv2.drawKeypoints(gray2, pts2, None, flags=4)

plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

plt.show()


# FLANN parameters
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 5)

search_params = dict(checks = 50)

# Convert to float32
descriptors1 = np.float32(kp1.compute(gray1, pts1)[1])
descriptors2 = np.float32(kp2.compute(gray2, pts2)[1])

# Create FLANN object
FLANN = cv2.FlannBasedMatcher(indexParams = index_params,
                             searchParams = search_params)

# Matching descriptor vectors using FLANN Matcher
matches = FLANN.knnMatch(queryDescriptors = descriptors1,
                         trainDescriptors = descriptors2,
                         k = 2)

# Lowe's ratio test
ratio_thresh = 0.7

# "Good" matches
good_matches = []

# Filter matches
for m, n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# Draw only "good" matches
output = cv2.drawMatches(img1 = img1,
                        keypoints1 = pts1,
                        img2 = img2,
                        keypoints2 = pts2,
                        matches1to2 = good_matches,
                        outImg = None,
                        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(output)
plt.axis('off')
plt.title('Good Matches for ' + sys.argv[1].upper())
plt.show()