import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import os

class HandControlMode:
    def __init__(self):
        # Mediapipe Tasks API setup
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the video mode:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hand_landmarker.task')
        
        # Check if model exists
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model file not found at {model_path}. Please run download_model.py")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.landmarker = HandLandmarker.create_from_options(options)
        
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 5
        self.last_click_time = 0
        self.timestamp_ms = 0

    def process(self, frame):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp (simulated for simplicity, ideally should use real frame timestamp)
        self.timestamp_ms += int(1000/30) # Assuming 30fps
        
        # Detect hands
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw landmarks (Manual drawing since we don't have drawing_utils compatible with Tasks result directly easily available without conversion, 
                # but we can implement a simple drawer or just use the coordinates)
                
                # Get landmarks
                # Index finger tip (ID 8)
                index_x = int(hand_landmarks[8].x * w)
                index_y = int(hand_landmarks[8].y * h)
                
                # Thumb tip (ID 4)
                thumb_x = int(hand_landmarks[4].x * w)
                thumb_y = int(hand_landmarks[4].y * h)

                # Draw simple landmarks
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (255, 0, 255), 3)

                # Convert to screen coordinates with smoothing
                # Simple smoothing
                curr_x = np.interp(index_x, (0, w), (0, self.screen_w))
                curr_y = np.interp(index_y, (0, h), (0, self.screen_h))
                
                curr_x = self.prev_x + (curr_x - self.prev_x) / self.smoothening
                curr_y = self.prev_y + (curr_y - self.prev_y) / self.smoothening

                self.prev_x, self.prev_y = curr_x, curr_y

                # Move Mouse
                try:
                     pyautogui.moveTo(curr_x, curr_y)
                except pyautogui.FailSafeException:
                     pass
                
                # Check for click (distance between index and thumb)
                # Normalized distance might be better, but pixel distance is what we have
                distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
                
                if distance < 30: # Threshold for click
                    cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
                    if time.time() - self.last_click_time > 0.5: # Debounce
                        pyautogui.click()
                        self.last_click_time = time.time()

        return frame
