import cv2
import numpy as np

def calculate_mse(img1, img2):
        #Convertir en pixels
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # MSE
        err = np.sum((img1 - img2) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        return err
#Charger des images
img1 = cv2.imread('/Users/ducnguyenanh/Downloads/iloveimg-converted (1)/IMG_3043.jpg')
img2 = cv2.imread('/Users/ducnguyenanh/Downloads/iloveimg-converted (1)/IMG_3044.jpg')

if img1 is None or img2 is None:
    #Asssurer que les 2 images sont trouvables
    print("Image not found.")
else:
    #Comparaison des images
    print(f"Size of image 1: {img1.shape}")
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    result = calculate_mse(img1, img2)
    print(f"MSE between images: {result}")


