import cv2
import numpy as np

def process_flower_image(img_bgr):
    """
    Traite une image de fleur (numpy.ndarray) pour segmenter la partie où il y a la fleur.

    Paramètre:
        img_bgr (numpy.ndarray): L'image source : NumPy array (BGR, format standard OpenCV).
        show_steps (bool): Si True, affiche les étapes de traitement avec matplotlib.

    Returns:
        tuple: (cropped_img, original_bbox) where cropped_img is the segmented and cropped flower image
               and original_bbox is a tuple (min_row, min_col, max_row, max_col) of the detected flower's bounding box in the original image.
               Returns (None, None) if no flower is detected or processing fails.
    """
    # 1. Conversions de base
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Rouge : On resserre la teinte et on demande une couleur très vive (S=100)
    mask_rouge = cv2.bitwise_or(
        cv2.inRange(hsv_img, np.array([0, 180, 120]), np.array([8, 255, 255])),
        cv2.inRange(hsv_img, np.array([170, 180, 120]), np.array([180, 255, 255]))
    )
    # Jaune : On monte H_min à 22 pour éviter l'orange/chair de la peau
    # Et on augmente S_min pour ne prendre que du jaune très saturé
    mask_jaune = cv2.inRange(hsv_img, np.array([15, 130, 150]), np.array([30, 255, 255]))
    
    # Bleu et Violet : On commence plus loin du vert (H=105)
    mask_bleu_violet = cv2.inRange(hsv_img, np.array([105, 100, 80]), np.array([155, 255, 255]))
    
    # Rose : Teinte située entre le violet et le rouge (Magenta/Rose vif)
    mask_rose = cv2.inRange(hsv_img, np.array([145, 0, 80]), np.array([255, 255, 255]))

    # Blanc : On demande une luminosité extrême (V=215) et une saturation très faible (S<35)
    mask_blanc = cv2.inRange(hsv_img, np.array([0, 0, 215]), np.array([180, 35, 255]))

    # Fusion de toutes les couleurs de fleurs détectées
    color_mask = cv2.bitwise_or(mask_rouge, cv2.bitwise_or(mask_jaune, cv2.bitwise_or(mask_bleu_violet, cv2.bitwise_or(mask_rose, mask_blanc))))
    
    # Nettoyage morphologique : on enlève les petits points (bruit) qui resteraient et on li les bouts proches ensuite
    # Taille du kernel adaptative à la résolution
    h, w = color_mask.shape[:2]
    k_size_open = max(3, int(max(h, w) * 0.005)) # 1.2% de la diagonale/dimension max (bien pour casser les tiges)
    k_size_close = max(3, int(max(h, w) * 0.025)) # 2% de la diagonale/dimension max

    kernel_open = np.ones((k_size_open, k_size_open), np.uint8)
    kernel_close = np.ones((k_size_close, k_size_close), np.uint8)

    mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)  # Supprime le bruit (petits points blancs)
    mask_after_open = mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)  # Bouche les trous (zones noires internes)

    # 5. Identification des régions et sélection de la fleur
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    best_score = -1
    original_bbox = None

    for i in range(1, num_labels): # 0 est le background
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        if area > best_score:
            best_score = area
            original_bbox = (y, x, y + h, x + w) # format (minr, minc, maxr, maxc)

    if original_bbox is None:
        return None

    cropped_img = rgb_img[original_bbox[0]:original_bbox[2], original_bbox[1]:original_bbox[3]]

    if cropped_img is not None:
        return cropped_img
    
    print(f"Aucune région détectée dans l'image.")
    return None
  
