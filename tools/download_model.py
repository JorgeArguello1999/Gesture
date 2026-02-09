import urllib.request
import os

# Ensure models directory exists
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)

url_hand = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
hand_model_path = os.path.join(models_dir, "hand_landmarker.task")
urllib.request.urlretrieve(url_hand, hand_model_path)
print(f"Model downloaded to {hand_model_path}")

url_face = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
face_model_path = os.path.join(models_dir, "face_landmarker.task")
urllib.request.urlretrieve(url_face, face_model_path)
print(f"Model downloaded to {face_model_path}")
