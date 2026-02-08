import cv2
import mediapipe as mp
import numpy as np
import os
import pyautogui
import time

class EyeControlMode:
    def __init__(self):
        # We need Face Landmarker, which provides 478 landmarks including iris
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'face_landmarker.task')
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}. Please run tools/download_model.py")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True)
            
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Smoothening
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 10 # Strong smoothing for gaze
        
        # Calibrating "Center" gaze
        # Ideally we'd calibrate, but let's assume looking straight = center screen
        # We use nose tip as anchor and iris relative position? 
        # Actually, for "Follow Eyes" robustly without per-user calib, 
        # combining Head Pose (nose) + Eye Gaze is best.
        # Let's start with Head Pose (Nose) as "Gaze" proxy because it's much more stable for mouse control
        # than raw eye tracking which jitters a lot.
        
        self.active = False # Toggle via facial gesture?

    def process(self, frame):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += int(1000/30)
        
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            
            # --- GAZE / HEAD TRACKING LOGIC ---
            # Landmark 1 is nose tip.
            # Landmark 468 is Left Iris Center, 473 is Right Iris Center
            
            nose_tip = landmarks[1]
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            
            # Using Nose Tip for broad movement (Head Pointing)
            # This is standard "Gaze Mouse" for accessibility without eye trackers
            
            # Define Active Region (Center of screen)
            margin_x = w * 0.4
            margin_y = h * 0.4
            
            # Draw Guide Box
            cv2.rectangle(frame, (int(margin_x), int(margin_y)), 
                                 (int(w - margin_x), int(h - margin_y)), (255, 0, 0), 1)

            # Map nose position to screen
            target_x = np.interp(nose_tip.x * w, (margin_x, w - margin_x), (0, self.screen_w))
            target_y = np.interp(nose_tip.y * h, (margin_y, h - margin_y), (0, self.screen_h))
            
            # Smoothing
            curr_x = self.prev_x + (target_x - self.prev_x) / self.smoothening
            curr_y = self.prev_y + (target_y - self.prev_y) / self.smoothening
            
            self.prev_x, self.prev_y = curr_x, curr_y
            
            # Visual Feedback
            cv2.circle(frame, (int(nose_tip.x * w), int(nose_tip.y * h)), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Eye/Head Control", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Move mouse
            try:
                pyautogui.moveTo(curr_x, curr_y)
            except:
                pass
                
            # --- BLINK DETECTION FOR CLICK? ---
            # Or simplified: just cursor movement for now as requested.
            
        return frame
