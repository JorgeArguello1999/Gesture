import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os

class FaceDetectionMode:
    def __init__(self):
        # Model Path
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_path, 'models', 'face_landmarker.task')

        # Check if model exists
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Face model not found at {model_path}")

        # Initialize MediaPipe Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Connections for manual drawing
        # Since mp.solutions is not available in this environment, we will skip
        # the full tessellation lines and just draw the landmarks as a point cloud.
        self.connections = [] 

    def process(self, frame):
        h, w, c = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect Face
        detection_result = self.landmarker.detect(mp_image)
        
        # Visualize
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                
                # Manual Drawing of Mesh (Point Cloud)
                points = []
                for lm in face_landmarks:
                    points.append((int(lm.x * w), int(lm.y * h)))
                
                # Draw Key Points as Green Dots (Tech Cloud)
                for pt in points:
                    cv2.circle(frame, pt, 1, (0, 255, 0), -1)
                
                # Draw Key Points (optional, maybe too crowded)
                # for pt in points:
                #    cv2.circle(frame, pt, 1, (0, 255, 255), -1)

                # Draw Bounding Box
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                
                x_min = min(x_values)
                y_min = min(y_values)
                x_max = max(x_values)
                y_max = max(y_values)
                
                # Add margin
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                # Draw Tech Box
                # Corner brackets instead of full rect? Or full rect.
                # Full rect for now + HUD elements.
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Tech Label
                cv2.putText(frame, f"TARGET LOCKED", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Mock Analysis Data
                cv2.putText(frame, f"CONFIDENCE: 99.8%", (x_max + 10, y_min + 20),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, f"ID: 8472-A", (x_max + 10, y_min + 40),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return frame
