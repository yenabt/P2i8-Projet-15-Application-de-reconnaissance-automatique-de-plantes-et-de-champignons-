import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, erosion 
from scipy import ndimage as ndi
import numpy as np
import os


def process_flower_image(img_array):
    """
    Traite une image de fleur (numpy.ndarray) pour segmenter la partie où il y a la fleur.

    Paramètre:
        img_array (numpy.ndarray): L'image source : NumPy array (RGB).

    Returns:
        L'image segmenté de la fleur (numpy.ndarray)
    """
    if img_array is None:
        print("Error: Pas d'image (image = None)")
        return None

    # Ensure image is RGB if it's grayscale or RGBA
    if img_array.ndim == 2:
        img = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] == 4: # RGBA to RGB
        img = img_array[:, :, :3]
    else:
        img = img_array # Assume it's already RGB

        # 1. Préparation des canaux : niveaux de gris et saturation
        gray_img = rgb2gray(img)
        hsv_img = rgb2hsv(img)
        saturation = hsv_img[:, :, 1]
        brightness = hsv_img[:, :, 2]

        # 2. Calcul des gradients sur les deux aspects
        grad_gray = sobel(gray_img)
        grad_sat = sobel(saturation)
        grad_combined = np.maximum(grad_gray, grad_sat)

        # 3. Création d'un masque de confiance (zones ni trop sombres, ni trop grises)
        # Ignore what is too dark (shadows) or too little saturated (dull earth/grass)
        confidence_mask = (saturation > np.mean(saturation)) & (brightness > 0.3)

        # 4. Masque final : Gradient combiné intersecté avec la zone de confiance
        seuil_grad = threshold_otsu(grad_combined)
        mask = (grad_combined > seuil_grad) & confidence_mask  # transforme l'image en binaire

        # 4. Nettoyage et unification de la zone de la fleur
        mask = dilation(mask, disk(2))
        mask = ndi.binary_fill_holes(mask)
        mask = erosion(mask, disk(3))
        mask = dilation(mask, disk(10))

        # 5. Identification des régions et sélection de la fleur
        labels = label(mask)
        regions = regionprops(labels, intensity_image=saturation)

        if regions:
            # saturation au carré pour donner de l'importance à la couleur 
            flower_region = max(regions, key=lambda r: r.area * (r.intensity_mean ** 2) if r.intensity_mean is not None else 0)
            minr, minc, maxr, maxc = flower_region.bbox
            
            # 5. Adaptive margin for cropping
            # Ensure margin calculation doesn't lead to negative dimensions or out-of-bounds access
            margin = int(max(maxr - minr, maxc - minc) * 0.1) + 100
            minr = max(0, minr - margin)
            minc = max(0, minc - margin)
            maxr = min(img.shape[0], maxr + margin)
            maxc = min(img.shape[1], maxc + margin)
            
            # Test si dimensions finales sont bonnes
            if minr >= maxr or minc >= maxc:
                print("Attention: Zone de l'image fausse.")
                return None

            cropped_img = img[minr:maxr, minc:maxc]
            return cropped_img
        else:
            print(f"Aucune région détectée dans l'image.")
            return None