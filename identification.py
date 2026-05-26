import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json

def extract_profile(img_bgr):
    """
    Extrait un profil hybride (couleur + forme) simplifié.
    Retourne un vecteur: [%R, %J, %B, %P, %W, AspectRatio, Extent]
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(8)

    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = h * w

    # Définition des masques (identiques à contour.py)
    masks = {
        "Rouge": cv2.bitwise_or(
            cv2.inRange(hsv_img, np.array([0, 180, 120]), np.array([8, 255, 255])),
        cv2.inRange(hsv_img, np.array([170, 180, 120]), np.array([180, 255, 255]))
        ),
        "Jaune": cv2.inRange(hsv_img, np.array([15, 130, 150]), np.array([30, 255, 255])),
        "Bleu": cv2.inRange(hsv_img, np.array([105, 100, 80]), np.array([125, 255, 255])),
        "Violet": cv2.inRange(hsv_img, np.array([125, 100, 80]), np.array([145, 255, 255])),
        "Rose": cv2.inRange(hsv_img, np.array([145, 40, 70]), np.array([175, 255, 255])),
        "Blanc": cv2.inRange(hsv_img, np.array([0, 0, 215]), np.array([180, 35, 255]))
    }

    profile = []
    # calcul profil de couleur
    for name in ["Rouge", "Jaune", "Bleu", "Violet", "Rose", "Blanc"]:
        count = cv2.countNonZero(masks[name])
        profile.append(count / total_pixels)

    # --- Nouvelles métriques de forme ---
    # Union de tous les masques pour définir la forme globale de la fleur
    all_flower_mask = np.bitwise_or.reduce(list(masks.values()))
    # 1. Rapport de forme (Normalisé pour être entre 0 et 1)
    aspect_ratio = min(w, h) / max(w, h)  # =1 pour une fleur ronde (segmentation carré)
    # 2. Étendue (Occupation du rectangle par la fleur)
    flower_area = cv2.countNonZero(all_flower_mask)
    extent = flower_area / total_pixels if total_pixels > 0 else 0

    profile.extend([aspect_ratio, extent])
    return np.array(profile)

def train_centroids(dataset_path):
    """Calcule le profil moyen pour chaque catégorie de fleurs."""
    centroids = {}
    print("Calcul des profils de référence...")
    
    for category in os.listdir(dataset_path):
        cat_path = os.path.join(dataset_path, category)
        if not os.path.isdir(cat_path):
            continue
            
        profiles = []
        for file in os.listdir(cat_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(cat_path, file))
                if img is not None:
                    # BGR (attendu par extract_profile) pas besoin de conversion
                    profiles.append(extract_profile(img))
        
        if profiles:
            centroids[category] = np.mean(profiles, axis=0)
            print(f"  {category} : Profil calculé.")
            
    return centroids

def save_centroids(centroids, filepath):
    """Sauvegarde les centroïdes dans un fichier JSON."""
    # On convertit les arrays numpy en listes pour qu'ils soient sérialisables en JSON
    serializable = {k: v.tolist() for k, v in centroids.items()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=4, ensure_ascii=False)

def load_centroids(filepath):
    """Charge les centroïdes depuis un fichier JSON."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # On convertit les listes lues en arrays numpy pour les calculs de distance
    return {k: np.array(v) for k, v in data.items()}

def predict_fast(img_bgr, centroids):
    """Prédit la catégorie en trouvant le centroïde le plus proche."""
    # extract_profile attend du BGR.
    current_profile = extract_profile(img_bgr)
    
    best_dist = float('inf')
    best_cat = "Inconnu"
    
    for category, avg_profile in centroids.items():
        # Distance Euclidienne
        dist = np.linalg.norm(current_profile - avg_profile)
        if dist < best_dist:
            best_dist = dist
            best_cat = category
            
    return best_cat, (1 - best_dist) * 100, current_profile

def plot_color_profiles(centroids, test_profile, test_label):
    """Affiche les graphiques des profils de couleur pour comparaison."""
    categories = list(centroids.keys())
    n_cats = len(categories)
    n_plots = n_cats + 1
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    metric_names = ["Rouge", "Jaune", "Bleu", "Violet", "Rose", "Blanc", "Rapport", "Etendue"]
    plot_colors = ['red', 'yellow', 'blue', 'purple', 'hotpink', 'lightgrey', 'green', 'orange']

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharey=True)
    axes = axes.flatten()
    
    # Affichage des centroïdes (références)
    for i, cat in enumerate(categories):
        axes[i].bar(metric_names, centroids[cat], color=plot_colors, edgecolor='black')
        axes[i].set_title(f"Ref: {cat}")
        axes[i].tick_params(axis='x', rotation=45)

    # Affichage du profil de l'image de test
    axes[n_cats].bar(metric_names, test_profile, color=plot_colors, edgecolor='black', linewidth=2)
    axes[n_cats].set_title(f"TEST\n(Pred: {test_label})", fontweight='bold')
    axes[n_cats].tick_params(axis='x', rotation=45)

    # Cacher les axes vides
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    plt.ylim(0, 1.0) # On affiche de 0 à 100%
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Laisser de la place pour le titre
    plt.suptitle("Comparaison des Profils de Couleur (Pourcentages de pixels par masque)", fontsize=16)
    plt.show()

class FlowerClassifier:
    """
    Interface simplifiée pour l'application Kivy.
    Charge les données au démarrage pour des prédictions instantanées.
    """
    def __init__(self, cache_filename="centroids_cache.json"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_path = os.path.join(self.base_dir, cache_filename)
        self.centroids = load_centroids(self.cache_path)
        
        if self.centroids is None:
            print(f"Avertissement : Cache '{cache_filename}' introuvable. "
                  "Lancez identification.py pour l'entraîner.")

    def identify(self, img_bgr):
        """Prend une image BGR (OpenCV) et retourne (nom_fleur, confiance)."""
        if self.centroids is None:
            return "Modèle non chargé", 0.0
        
        name, confidence, _ = predict_fast(img_bgr, self.centroids)
        return name, confidence

if __name__ == "__main__":
    # Chemins
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "data_set_segmentee3")
    
    if not os.path.exists(dataset_path):
        print(f"Erreur : Dossier non trouvé {dataset_path}")
        sys.exit()

    cache_file = os.path.join(base_dir, "centroids_cache.json")
    # Changez cette variable à True si vous modifiez vos images ou vos masques HSV
    force_recompute = True 

    # 1. Chargement ou calcul des profils de référence
    if not force_recompute and os.path.exists(cache_file):
        print(f"Chargement des profils de référence depuis : {cache_file}")
        ref_centroids = load_centroids(cache_file)
    else:
        ref_centroids = train_centroids(dataset_path)
        save_centroids(ref_centroids, cache_file)
        print(f"Profils mis à jour et sauvegardés dans {cache_file}")

    # 2. Test sur une image
    test_img_path = r"Projet\reconnaissance_images\data_set_segmentee\marguerite-paquerette\Copie de IMG_20260512_101337.jpg"
    test_img = cv2.imread(test_img_path)

    if test_img is not None:
        fleur, confiance, profil_test = predict_fast(test_img, ref_centroids)
        print(f"\n--- Résultat Rapide ---")
        print(f"Image : {os.path.basename(test_img_path)}")
        print(f"Catégorie prédite : {fleur}")
        print(f"Indice de proximité : {confiance:.2f}%")
        
        # Affichage visuel pour débogage
        plot_color_profiles(ref_centroids, profil_test, fleur)