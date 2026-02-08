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
            num_hands=2, # Enable 2 hands detection
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.landmarker = HandLandmarker.create_from_options(options)
        
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 5
        self.timestamp_ms = 0
        
        # State for click dragging
        self.is_left_clicking = False
        self.last_right_click_time = 0
        
        # Configure pyautogui for speed
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False # Be careful with this, but prevents some interruptions

    def process(self, frame):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp 
        self.timestamp_ms += int(1000/30) 
        
        # Detect hands
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        # Visual feedback instructions
        cv2.putText(frame, "Right Hand (Movement)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Left Hand (Actions)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "- Index+Thumb: Left Click/Drag", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "- Middle+Thumb: Right Click", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[i][0]
                category_name = handedness.category_name
                
                self._draw_landmarks(frame, hand_landmarks)

                # MIRROR MODE HANDLING:
                # In a mirrored view (selfie camera), the physical Right Hand appears on the Right side of the screen.
                # MediaPipe detects this geometry as a "Left" hand because it's a mirror image.
                # So we swap the logic: 
                # MP "Left" -> It's the User's Right Hand (Mouse Movement)
                # MP "Right" -> It's the User's Left Hand (Clicks)
                
                if category_name == "Left":  # This is physically Right Hand in mirror mode
                    self._handle_right_hand(hand_landmarks, frame, w, h)
                    cv2.putText(frame, "Right Hand", (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h) - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                elif category_name == "Right": # This is physically Left Hand in mirror mode
                    self._handle_left_hand(hand_landmarks, frame, w, h)
                    cv2.putText(frame, "Left Hand", (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h) - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame

    def _draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        for pt in landmarks:
            x = int(pt.x * w)
            y = int(pt.y * h)
            cv2.circle(frame, (x, y), 5, (200, 200, 200), -1)

    def _handle_right_hand(self, landmarks, frame, w, h):
        """Right Hand: Controls Mouse Movement (Mapped from MP 'Left' in mirror mode)"""
        # Index finger tip (ID 8)
        index_x = int(landmarks[8].x * w)
        index_y = int(landmarks[8].y * h)
        
        cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED) 

        # Convert to screen coordinates with smoothing
        curr_x = np.interp(index_x, (0, w), (0, self.screen_w))
        curr_y = np.interp(index_y, (0, h), (0, self.screen_h))
        
        # Increase smoothening for better stability
        curr_x = self.prev_x + (curr_x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (curr_y - self.prev_y) / self.smoothening

        self.prev_x, self.prev_y = curr_x, curr_y

        try:
             pyautogui.moveTo(curr_x, curr_y)
        except pyautogui.FailSafeException:
             pass

    def _handle_left_hand(self, landmarks, frame, w, h):
        """Left Hand: Controls Clicks (Mapped from MP 'Right' in mirror mode)"""
        # Thumb tip (ID 4)
        thumb_x = int(landmarks[4].x * w)
        thumb_y = int(landmarks[4].y * h)
        
        # Index tip (ID 8)
        index_x = int(landmarks[8].x * w)
        index_y = int(landmarks[8].y * h)
        
        # Middle tip (ID 12)
        middle_x = int(landmarks[12].x * w)
        middle_y = int(landmarks[12].y * h)

        # --- Gesture 1: Left Click / Drag (Index + Thumb) ---
        dist_left = np.hypot(index_x - thumb_x, index_y - thumb_y)
        if dist_left < 30:
            cv2.circle(frame, (index_x, index_y), 15, (0, 255, 255), cv2.FILLED) 
            if not self.is_left_clicking:
                pyautogui.mouseDown()
                self.is_left_clicking = True
            cv2.putText(frame, "Dragging", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            if self.is_left_clicking:
                pyautogui.mouseUp()
                self.is_left_clicking = False

        # --- Gesture 2: Right Click (Middle + Thumb) ---
        dist_right = np.hypot(middle_x - thumb_x, middle_y - thumb_y)
        if dist_right < 30:
            cv2.circle(frame, (middle_x, middle_y), 15, (0, 0, 255), cv2.FILLED)
            # Debounce right click, but don't block
            if time.time() - self.last_right_click_time > 0.5:
                pyautogui.click(button='right')
                self.last_right_click_time = time.time()
                cv2.putText(frame, "Right Click", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
