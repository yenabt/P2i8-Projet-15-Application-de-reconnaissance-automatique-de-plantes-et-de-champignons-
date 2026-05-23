from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Rotate, PushMatrix, PopMatrix
from android.permissions import request_permissions, Permission
import os # For path manipulation
import sys # For modifying sys.path to import local modules
import cv2 # To load and save images for processing
import numpy as np

# Add the directory containing contour.py to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'reconnaissance_images'))
from contour import process_flower_image # Import the function

class CameraApp(App):
    def build(self):
        request_permissions([Permission.CAMERA])

        layout = BoxLayout(orientation='vertical')

        self.camera = Camera(resolution=(640, 480), play=True)

        # Correction de la rotation de 90 degrés
        with self.camera.canvas.before:
            PushMatrix()
            self.rot = Rotate(angle=-90, origin=self.camera.center)
        with self.camera.canvas.after:
            PopMatrix()
        self.camera.bind(pos=self._update_camera_rotation, size=self._update_camera_rotation)

        # Widget pour afficher la dernière photo prise
        self.last_photo = Image()

        btn = Button(text="Capture et Traitement")
        btn.bind(on_press=self.capture_and_process)

        layout.add_widget(self.camera)
        layout.add_widget(self.last_photo)
        layout.add_widget(btn)

        return layout
    
    def _update_camera_rotation(self, instance, value):
        self.rot.origin = instance.center

    def capture_and_process(self, instance):
        original_photo_path = os.path.join(self.user_data_dir, "original_photo.png")
        processed_photo_path = os.path.join(self.user_data_dir, "processed_photo.png")

        self.camera.export_to_png(original_photo_path)
        print("Original photo saved:", original_photo_path)

        try:
            # OpenCV charge en BGR par défaut, on convertit en RGB pour l'algorithme
            img_bgr = cv2.imread(original_photo_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Impossible de lire l'image à {original_photo_path}")
        except Exception as e:
            print(f"Error loading captured image {original_photo_path}: {e}")
            self.last_photo.source = original_photo_path # Fallback
            self.last_photo.reload()
            return

        # Gère le retour : soit l'image seule, soit le tuple (image, bbox)
        result = process_flower_image(img_bgr)
        processed_img_array = result

        if processed_img_array is not None:
            if processed_img_array.dtype != np.uint8:
                processed_img_array = (processed_img_array * 255).astype(np.uint8)
            
            # Conversion RGB vers BGR avant de sauvegarder avec OpenCV
            img_bgr_out = cv2.cvtColor(processed_img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(processed_photo_path, img_bgr_out)
            print("Processed photo saved:", processed_photo_path)
            self.last_photo.source = processed_photo_path
        else:
            print("No flower detected or processing failed. Displaying original photo.")
            self.last_photo.source = original_photo_path
        
        self.last_photo.reload()

CameraApp().run()
