import os
import numpy as np
import matplotlib.pyplot as plt
from contour import process_flower_image
import cv2

NAMED_COLORS = {
    "Rouge":   [53, 80, 67],
    "Orange":  [63, 43, 74],
    "Jaune":   [97, -21, 94],
    "Vert":    [36, -43, 41],
    "Bleu":    [32, 79, -107],
    "Violet":  [29, 58, -36],
    "Rose":    [66, 64, 17],
    "Blanc":   [100, 0, 0],
    "Magenta": [60, 93, -60],
    "Cyan":    [91, -48, -14],
}

def segment_flower(fleur_bgr):
    """
    Segmente la fleur dans une image en isolant le premier plan du fond.

    Applique un flou gaussien, une analyse HSV (saturation + valeur),
    un seuillage d'Otsu, puis l'algorithme de Watershed pour délimiter
    précisément la région de la fleur.

    Paramètres
    ----------
    fleur_bgr : numpy.ndarray
        Image de la fleur au format BGR (tel que retourné par cv2.imread).

    Retourne
    --------
    fg_mask : numpy.ndarray
        Masque binaire (uint8) de même taille que l'image d'entrée,
        où 255 correspond aux pixels appartenant à la fleur et 0 au fond.
    """        
    blurred = cv2.GaussianBlur(fleur_bgr, (31, 31), 0)

    fleur_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    fleur_S = fleur_hsv[:, :, 1]
    fleur_V = fleur_hsv[:, :, 2]

    _, thresh_s = cv2.threshold(fleur_S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_v = cv2.threshold(fleur_V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rough = cv2.bitwise_or(thresh_s, thresh_v)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    sure_fg = cv2.erode(rough, kernel, iterations=2)
    sure_bg = cv2.dilate(rough, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(fleur_bgr, markers)

    flower_label = markers[markers.shape[0] // 2, markers.shape[1] // 2]
    fg_mask = np.uint8(markers == flower_label) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_small)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_mask = np.zeros_like(fg_mask)
    cv2.drawContours(fg_mask, contours, -1, 255, thickness=cv2.FILLED)

    return fg_mask


def detect_color(fleur_bgr, fg_mask):
    """
    Détecte la couleur dominante de la fleur parmi une palette de couleurs nommées.

    Convertit les pixels du premier plan en espace colorimétrique CIE L*a*b*,
    calcule leur moyenne, puis retourne le nom de la couleur la plus proche
    par distance euclidienne dans NAMED_COLORS.

    Paramètres
    ----------
    fleur_bgr : numpy.ndarray
        Image de la fleur au format BGR.
    fg_mask : numpy.ndarray
        Masque binaire (uint8) indiquant les pixels de la fleur (255 = fleur).

    Retourne
    --------
    str
        Nom de la couleur la plus proche parmi les clés de NAMED_COLORS
        (ex. : "Rouge", "Bleu", "Jaune", etc.).
    """
    fleur_lab = cv2.cvtColor(fleur_bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    fg_pixels = fleur_lab[fg_mask.ravel() == 255].astype(np.float32)
    fg_pixels[:, 0] = fg_pixels[:, 0] * 100 / 255
    fg_pixels[:, 1] = fg_pixels[:, 1] - 128
    fg_pixels[:, 2] = fg_pixels[:, 2] - 128

    mean_lab = fg_pixels.mean(axis=0)
    return min(NAMED_COLORS, key=lambda c: np.linalg.norm(mean_lab - NAMED_COLORS[c]))

def color_and_image(img):
    """
    Pipeline complet : charge l'image, segmente la fleur et détecte sa couleur.

    Lit l'image 'test4.jpeg' depuis le répertoire du script, applique le
    prétraitement de contour via process_flower_image, segmente la fleur,
    identifie sa couleur dominante, puis retourne l'image recadrée sans fond.

    Paramètres
    ----------
    img : str
        Nom du fichier image (non utilisé directement — l'image est chargée
        en dur depuis 'test4.jpeg' dans le répertoire courant du script).

    Retourne
    --------
    color : str
        Nom de la couleur dominante détectée (ex. : "Rose", "Violet").
    fleur_no_bg : numpy.ndarray
        Image de la fleur sans fond (pixels hors masque mis à zéro),
        au format RGB.
    """
    fleur_bgr = cv2.imread(os.path.join(script_dir, 'test4.jpeg'))
    contour = process_flower_image(fleur_bgr)
    fleur_rgb = cv2.bilateralFilter(cv2.cvtColor(fleur_bgr, cv2.COLOR_BGR2RGB), 9, 75, 75)

    fg_mask = segment_flower(contour)
    color = detect_color(contour, fg_mask)

    fleur_no_bg = contour.copy()
    fleur_no_bg[fg_mask == 0] = 0

    return color, fleur_no_bg



script_dir = os.path.dirname(os.path.abspath(__file__))
fleur = 'test4.jpeg'
color, img = color_and_image(fleur)

print(color)

plt.imshow(img)
plt.show()