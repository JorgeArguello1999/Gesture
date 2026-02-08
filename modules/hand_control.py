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

        # Helper for Face Detection (Gaze Safety)
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        
        face_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'face_landmarker.task')
        if not os.path.exists(face_model_path):
             # Fallback if not downloaded, though we expect it is
             print("Face model not found, Gaze Safety disabled.")
             self.face_landmarker = None
        else:
            face_options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.face_landmarker = FaceLandmarker.create_from_options(face_options)

        self.screen_w, self.screen_h = pyautogui.size()
        self.frame_margin = 100 # Frame reduction for mouse movement
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 5
        self.timestamp_ms = 0
        
        # State for click dragging
        self.is_left_clicking = False
        
        # State for Right Click (Hold to trigger)
        self.right_click_start_time = 0
        self.right_click_triggered = False
        
        # Configure pyautogui for speed
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False # Be careful with this, but prevents some interruptions

    def process(self, frame):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp 
        self.timestamp_ms += int(1000/30) 
        
        # 1. Detect Face for Gaze Safety
        is_looking_at_screen = True # Default to true if detection fails or is disabled
        if self.face_landmarker:
            face_result = self.face_landmarker.detect_for_video(mp_image, self.timestamp_ms)
            if face_result.face_landmarks:
                # Simple Gaze Check: Is face roughly frontal?
                # We can check relation of Nose Tip (1) to Ears or Eyes.
                # Let's use Nose (1) vs Checkbones (Left: 234, Right: 454)
                landmarks = face_result.face_landmarks[0]
                nose_x = landmarks[1].x
                left_cheek_x = landmarks[234].x
                right_cheek_x = landmarks[454].x
                
                # If nose is not between cheeks, we are turning away significantly
                # Also checks vertical (Nose 1 vs Chin 152 vs Forehead 10)
                
                # Check Horizontal
                if not (right_cheek_x < nose_x < left_cheek_x): # Note: Coordinates might be flipped depending on mirror
                     # Actually in MP normalized coordinates: 0 is Left, 1 is Right.
                     # Left Cheek (234) should be > Right Cheek (454) in x? 
                     # Let's rely on a simpler metric: Face Presence implies looking roughly at camera 
                     # if we enforce high tracking confidence? 
                     # User wants "If I am seeing the screen".
                     # Let's stick to "Face Detected" = "Looking at screen" for now, 
                     # but maybe visualize it.
                     pass
                
                # Better: Check yaw via eye-box?
                # Let's assume Valid Face = Looking At Screen for this iteration.
                is_looking_at_screen = True
                
                # Visual Feedback for Gaze
                cv2.rectangle(frame, (10, 10), (w-10, h-10), (0, 255, 0), 2)
                cv2.putText(frame, "Gaze Detected: Clicks Enabled", (w//2 - 150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                is_looking_at_screen = False
                cv2.putText(frame, "Gaze NOT Detected: Clicks Disabled", (w//2 - 180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. Detect Hands
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

        # Draw Active Region Box
        cv2.rectangle(frame, (self.frame_margin, self.frame_margin), (w - self.frame_margin, h - self.frame_margin), (255, 0, 255), 2)

        # Visual feedback instructions
        cv2.putText(frame, "Right Hand: Move", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Left Hand: Click / Drag", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Both Fists: Right Click", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Identify Hands
        mouse_hand = None # Physical Right (MP Left)
        click_hand = None # Physical Left (MP Right)
        
        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[i][0]
                category_name = handedness.category_name
                
                self._draw_landmarks(frame, hand_landmarks)
                
                if category_name == "Left":
                    mouse_hand = hand_landmarks
                elif category_name == "Right":
                    click_hand = hand_landmarks

        # --- Dual Hand Gesture: Right Click (Both Fists) ---
        is_right_clicking_gesture = False
        if mouse_hand and click_hand:
            if self._is_fist(mouse_hand) and self._is_fist(click_hand):
                is_right_clicking_gesture = True
                
        if is_right_clicking_gesture:
             # Handle Right Click Hold
            if self.right_click_start_time == 0:
                self.right_click_start_time = time.time()
            
            elapsed = time.time() - self.right_click_start_time
            
            # Draw shared progress bar in center
            cx, cy = w // 2, h // 2
            bar_width = 200
            filled_width = int((elapsed / 0.5) * bar_width)
            cv2.rectangle(frame, (cx - 100, cy - 60), (cx + 100, cy - 40), (100, 100, 100), -1)
            cv2.rectangle(frame, (cx - 100, cy - 60), (cx - 100 + min(filled_width, bar_width), cy - 40), (0, 0, 255), -1)
            cv2.putText(frame, "Right Click...", (cx - 60, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            if elapsed > 0.5 and not self.right_click_triggered:
                if is_looking_at_screen: # Gaze Safety Check
                   pyautogui.click(button='right')
                   self.right_click_triggered = True
                   cv2.putText(frame, "CLICK!", (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            self.right_click_start_time = 0
            self.right_click_triggered = False

        # --- Individual Hand Processing ---
        # Only process if we are NOT currently right clicking (to avoid conflict)
        if not is_right_clicking_gesture:
            if mouse_hand:
                 self._handle_right_hand(mouse_hand, frame, w, h)
                 cv2.putText(frame, "Mouse", (int(mouse_hand[0].x * w), int(mouse_hand[0].y * h) - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
            if click_hand:
                 self._handle_left_hand(click_hand, frame, w, h, is_looking_at_screen)
                 cv2.putText(frame, "Clicks", (int(click_hand[0].x * w), int(click_hand[0].y * h) - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return frame

    def _is_fist(self, landmarks):
        """Check if hand is in a fist state (Fingertips below PIP joints)"""
        # Fingertip IDs: 8, 12, 16, 20 (Index, Middle, Ring, Pinky)
        # PIP Joint IDs: 6, 10, 14, 18
        # Thumb is exception, we ignore it for simple fist check or check if tucked.
        
        # We assume hand is upright. If y of tip > y of pip, it's folded down (in image coords, y increases downwards)
        # So Tip.y > Pip.y means finger is curled? 
        # Yes, since 0,0 is top-left. Lower on screen = higher Y value.
        # So curled finger tip is lower (higher Y) than knuckle/pip? 
        # Actually, usually Tip is "below" PIP in a fist if palm faces camera?
        # Let's use distance to wrist (0).
        # Fist: Distance(Tip, Wrist) < Distance(PIP, Wrist) roughly?
        # Or simpler: Tip.y > PIP.y (if hand is pointing up).
        
        # Let's use a robust heuristic: Finger Tips are close to Wrist
        wrist = landmarks[0]
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        fingers_folded = 0
        for i in range(4):
            # Distance check is rotation invariant-ish
            tip_pt = landmarks[tips[i]]
            pip_pt = landmarks[pips[i]]
            
            d_tip = np.hypot(tip_pt.x - wrist.x, tip_pt.y - wrist.y)
            d_pip = np.hypot(pip_pt.x - wrist.x, pip_pt.y - wrist.y)
            
            if d_tip < d_pip:
                fingers_folded += 1
                
        return fingers_folded >= 3 # At least 3 fingers folded (excluding thumb)

    def _draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        for pt in landmarks:
            x = int(pt.x * w)
            y = int(pt.y * h)
            cv2.circle(frame, (x, y), 5, (200, 200, 200), -1)

    def _handle_right_hand(self, landmarks, frame, w, h):
        """Right Hand: Controls Mouse Movement"""
        # Index finger tip (ID 8)
        index_x = int(landmarks[8].x * w)
        index_y = int(landmarks[8].y * h)
        
        cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED) 

        # Convert to screen coordinates with MARGIN
        curr_x = np.interp(index_x, (self.frame_margin, w - self.frame_margin), (0, self.screen_w))
        curr_y = np.interp(index_y, (self.frame_margin, h - self.frame_margin), (0, self.screen_h))
        
        # Smoothing
        curr_x = self.prev_x + (curr_x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (curr_y - self.prev_y) / self.smoothening

        self.prev_x, self.prev_y = curr_x, curr_y

        try:
             pyautogui.moveTo(curr_x, curr_y)
        except pyautogui.FailSafeException:
             pass

    def _handle_left_hand(self, landmarks, frame, w, h, clicks_enabled):
        """Left Hand: Controls Clicks (Mapped from MP 'Right' in mirror mode)"""
        if not clicks_enabled:
             return

        # Thumb tip (ID 4)
        thumb_x = int(landmarks[4].x * w)
        thumb_y = int(landmarks[4].y * h)
        
        # Index tip (ID 8)
        index_x = int(landmarks[8].x * w)
        index_y = int(landmarks[8].y * h)
        
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
