import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_flower_image(img_bgr, show_steps=False):
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
    

    # 3. Création d'un masque de confiance
    # On définit des plages pour exclure le fond (vert et marron/terre)
    # En HSV OpenCV: H [0, 179], S [0, 255], V [0, 255].
    
    '''# FLEUR BLANCHE
    mask_vert = cv2.inRange(hsv_img, np.array([38, 35, 0]), np.array([95, 255, 255]))
    mask_marron = cv2.inRange(hsv_img, np.array([10, 30, 0]), np.array([18, 255, 255]))'''

    '''# FLEUR BLANCHE
    mask_vert = cv2.inRange(hsv_img, np.array([25, 35, 0]), np.array([95, 255, 255]))
    mask_marron = cv2.inRange(hsv_img, np.array([10, 30, 0]), np.array([25, 255, 150]))'''

    '''# FLEUR BLANCHE
    mask_vert = cv2.inRange(hsv_img, np.array([30, 45, 0]), np.array([90, 255, 250]))
    mask_marron = cv2.inRange(hsv_img, np.array([10, 40, 0]), np.array([25, 255, 200]))'''

    '''# FLEUR ROUGE BOFFFFFF
    mask_vert = cv2.inRange(hsv_img, np.array([25, 35, 0]), np.array([95, 255, 255]))
    mask_marron = cv2.inRange(hsv_img, np.array([10, 0, 20]), np.array([11, 255, 200]))

    # On combine les deux pour créer le masque des zones à exclure
    color_mask = cv2.bitwise_or(mask_vert, mask_marron)
    color_mask = cv2.bitwise_not(color_mask)'''

    # 3. Masque d'inclusion : On restreint les plages pour être plus sélectif
    # On augmente S_min et V_min pour ignorer le décor (souvent moins saturé et moins lumineux)
    
    # Rouge : On resserre la teinte et on demande une couleur très vive (S=100)
    mask_rouge = cv2.bitwise_or(
        cv2.inRange(hsv_img, np.array([0, 100, 70]), np.array([10, 255, 255])),
        cv2.inRange(hsv_img, np.array([160, 100, 70]), np.array([180, 255, 255]))
    )
    # Jaune : On monte H_min à 22 pour éviter l'orange/chair de la peau
    # Et on augmente S_min pour ne prendre que du jaune très saturé
    mask_jaune = cv2.inRange(hsv_img, np.array([15, 130, 150]), np.array([30, 255, 255]))
    
    # Bleu : On cible les tons bleus francs
    mask_bleu = cv2.inRange(hsv_img, np.array([105, 100, 80]), np.array([125, 255, 255]))

    # Violet : Spécifiquement pour les becs de grue (Erodium/Geranium)
    mask_violet = cv2.inRange(hsv_img, np.array([125, 100, 80]), np.array([145, 255, 255]))
    
    # Rose : Teinte située entre le violet et le rouge (Magenta/Rose vif)
    mask_rose = cv2.inRange(hsv_img, np.array([145, 40, 70]), np.array([175, 255, 255]))

    # Blanc : On demande une luminosité extrême (V=215) et une saturation très faible (S<35)
    mask_blanc = cv2.inRange(hsv_img, np.array([0, 0, 215]), np.array([180, 35, 255]))

    # Fusion de toutes les couleurs de fleurs détectées
    all_masks = [mask_rouge, mask_jaune, mask_bleu, mask_violet, mask_rose, mask_blanc]
    color_mask = np.bitwise_or.reduce(all_masks)
    
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
        return None, None

    cropped_img = img_bgr[original_bbox[0]:original_bbox[2], original_bbox[1]:original_bbox[3]]


    if show_steps:
        plt.figure(figsize=(15, 12))
        # Liste des étapes à afficher
        # images_to_show = [img, grad_8u, green_mask, mask_otsu, conf_mask, mask]
        images_to_show = [rgb_img, color_mask, mask_after_open, mask]

        titles = ["1. Image Originale", "2. Mask couleur", "3. Mask after open",  "4. Mask final"]
        
        for idx, (view, title) in enumerate(zip(images_to_show, titles)):
            plt.subplot(2, 3, idx + 1)
            plt.imshow(view, cmap='gray' if len(view.shape) == 2 else None)
            plt.title(title)
            plt.axis('off')
            
        if cropped_img is not None:
            plt.subplot(2, 3, 6)
            plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            plt.title("7. Résultat Recadré")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    if cropped_img is not None:
        return cropped_img, original_bbox
    
    print(f"Aucune région détectée dans l'image.")
    return None, None
    
def debug_color_masks(img_bgr):
    """
    Affiche chaque masque de couleur individuellement pour ajuster les seuils HSV.
    """
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Définition des masques (identiques à process_flower_image)
    masks = {
        "Rouge": cv2.bitwise_or(
            cv2.inRange(hsv_img, np.array([0, 120, 120]), np.array([8, 255, 255])),
            cv2.inRange(hsv_img, np.array([170, 120, 120]), np.array([180, 255, 255]))
        ),
        "Jaune": cv2.inRange(hsv_img, np.array([15, 130, 150]), np.array([35, 255, 255])),
        "Bleu": cv2.inRange(hsv_img, np.array([105, 100, 80]), np.array([125, 255, 255])),
        "Violet": cv2.inRange(hsv_img, np.array([125, 100, 80]), np.array([145, 255, 255])),
        "Rose": cv2.inRange(hsv_img, np.array([145, 40, 70]), np.array([175, 255, 255])),
        "Blanc": cv2.inRange(hsv_img, np.array([0, 0, 215]), np.array([180, 35, 255]))
    }

    plt.figure(figsize=(18, 10))
    
    # Image originale
    plt.subplot(2, 4, 1)
    plt.imshow(rgb_img)
    plt.title("Original RGB")
    plt.axis('off')

    # Affichage de chaque canal
    for i, (name, mask) in enumerate(masks.items()):
        # On applique le masque sur l'image RGB pour voir les couleurs réelles extraites
        result = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)
        
        plt.subplot(2, 4, i + 2)
        plt.imshow(result)
        # Calcul du pourcentage de l'image couvert par ce masque pour aider au réglage
        coverage = (np.sum(mask > 0) / mask.size) * 100
        plt.title(f"Masque {name}\n({coverage:.2f}% de l'image)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def affiche_resultat_segmentation(img_bgr):
    if img_bgr is not None:
        img_rgb =cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        cropped_rgb, bbox = process_flower_image(img_bgr, show_steps=False)
        if cropped_rgb is not None:
            plt.subplot(1,2,2)
            plt.imshow(cv2.cvtColor(cropped_rgb, cv2.COLOR_BGR2RGB))
            plt.title("Image segmenté")
            plt.axis('off')

            plt.subplot(1,2,1)
            plt.imshow(img_rgb)
            plt.title("Image d'origine")
            plt.axis('off')

if __name__ == "__main__":  # le test ne s'execute pas si on lance ce programme depuis un autre programme
    img_path = r'Projet\reconnaissance_images\Data Set Restreint\bec-de-grue\IMG_7961.jpg'
    # OpenCV charge en BGR par défaut
    img_bgr = cv2.imread(img_path)

    # affiche_resultat_segmentation(img_bgr)
    
    if img_bgr is not None:
        # Visualisation des masques de couleur pour le réglage
        debug_color_masks(img_bgr)
        
        cropped_rgb, bbox = process_flower_image(img_bgr, show_steps=True)
        
        if cropped_rgb is None:
            print("Échec de la segmentation.")
