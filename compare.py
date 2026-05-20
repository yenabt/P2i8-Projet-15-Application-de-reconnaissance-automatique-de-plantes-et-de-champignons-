import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def calculer_mse(img1, img2):
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # Le MSE
        err = np.sum((img1 - img2) ** 2)
        err /= float(img1.shape[0] * img1.shape[1])
        
        return err

img1 = cv2.imread('/Users/ducnguyenanh/Downloads/iloveimg-converted (1)/IMG_3043.jpg')
img2 = cv2.imread('/Users/ducnguyenanh/Downloads/iloveimg-converted (1)/IMG_3044.jpg')

def calculer_SSIM(img1, img2):
     #Convertir les images en grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #Calculer le SSIM
    index = ssim(img1_gray, img2_gray)
    return index

print(f"SSIM between images: {calculer_SSIM(img1, img2)}")


from sklearn.neighbors import KNeighborsClassifier


def pretraiter(img_path, target_size=(64, 64)):
    """
    param img_path: chemin vers l'image à prétraiter
    param target_size: la taille à laquelle redimensionner l'image (par défaut 64x64 pixels)
    Convertit une image en grayscale, la redimensionne à une taille standard, puis les pixels en un vecteur de coordonnées.
    """
    # grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # redimensionne
    img_resized = cv2.resize(img, target_size)
    
    # Flatten: transformer l'image 2D en un vecteur 1D de coordonnées (pixels)
    return img_resized.flatten()


#Le MAP: Un tableau de coordonnées (X) et un tableau de labels (y)
def map(dataset_path):
    """
    param dataset_path: chemin vers le dossier contenant les sous-dossiers de fleurs (ex: "Rose", "Tulipe", etc.)
    Looper à travers les dossiers pour créer la grille de coordonnées (X)
    et collecter les étiquettes de texte correspondantes (y).
    """
    X_map = []
    y_labels = []
    
    if not os.path.exists(dataset_path):
        # Si le chemin du dataset n'existe pas, afficher un message d'erreur et retourner des tableaux vides
        print(f"{dataset_path} n'existe pas. Veuillez vérifier le chemin d'accès.")
        return None, None

    # Scanner chaque dossier de fleur (Rose, Tulipe, etc.)
    for dossier_fleur in os.listdir(dataset_path):
        #Traverser le dataset_path
        #listdir: lister tous les objets qui sont actuellement dans le dossier dataset_path.
        #dossier_fleur: chaque dossier de fleur (Rose, Tulipe, etc.)
        
        dossier_fleur_chemin = os.path.join(dataset_path, dossier_fleur)
        #dossier_fleur_chemin: chemin complet vers le dossier de la fleur actuelle


        if os.path.isdir(dossier_fleur_chemin):
            #os.path.isdir: vérifier si le chemin est un dossier (et pas un fichier) pour assurer que nous traitons uniquement les dossiers de fleurs.
            for img in os.listdir(dossier_fleur_chemin):
                #listdir(dossier_fleur_chemin): lister tous les fichiers d'image dans le dossier de la fleur actuelle.

                img_chemin = os.path.join(dossier_fleur_chemin, img)
                #img_chemin: chemin complet vers l'image actuelle à traiter.
                
                # Prétraiter l'image et extraire les coordonnées en utilisant la fonction pretraiter en définie ci-dessus.
                coordinates = pretraiter(img_chemin)
                #pretraiter(img_chemin): convertir l'image en grayscale, la redimensionner à une taille standard, puis les pixels en un vecteur de coordonnées.

                if coordinates is not None:
                    #Verifie si les coord sont valides.

                    X_map.append(coordinates)
                    #On append les coords

                    y_labels.append(dossier_fleur) # Attacher les labels ("Rose", "Tulipe")
                    #On append les labels
                    
    return np.array(X_map), np.array(y_labels)


# Comparaison de deux images par KNN
if __name__ == "__main__":
    
    # CONFIGURATION (à faire pour chaque PC perso)
    DATASET_DIRECTOIRE = '/Users/ducnguyenanh/Downloads/my_flower_dataset/'
    IMAGE_CHEMIN = '/Users/ducnguyenanh/Downloads/iloveimg-converted (1)/IMG_3071.jpg'

    X_train, y_train = map(DATASET_DIRECTOIRE)
    
    if X_train is not None and len(X_train) > 0:
        print(f"Nb images: {X_train.shape[0]}")
        print(f"Dimensions: {X_train.shape[1]}")
        
        # Initialiser le KNN. On prendra K=3
        # 'euclidean' pour un calcul de distance dans les espaces de coordonnées d'images.
        k_value = 3
        knn = KNeighborsClassifier(n_neighbors=k_value, metric='euclidean')
        
        # Input les coords et les labels dans le modèle KNN
        knn.fit(X_train, y_train)
        print("KNN est entraîné.")
        
        # COMPARAISON
        X_image = pretraiter(IMAGE_CHEMIN)
        
        #si les coords sont valides
        if X_image is not None:
            
            # Redimensionne pour conformer le format attendu par KNN (1 échantillon, n features)
            X_image = X_image.reshape(1, -1)
            
            # Prediction de la classe de l'image mystère
            prediction = knn.predict(X_image)
            
            # Distance et indices des K voisins les plus proches
            distances, indices = knn.kneighbors(X_image)
            
            print(f"L'image est classée comme: {prediction[0]}")
            print(f"Distances aux voisins les plus proches: {distances[0]}")
            print(f"Leurs indexes dans le dataset original sont: {indices[0]}")
        else:
            print("Erreur: Ne peut pas charger le fichier image.")
    else:
        print("Echec de la construction du map.")
