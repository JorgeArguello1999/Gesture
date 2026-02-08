import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

class HandControlMode:
    def __init__(self):
        # Explicitly try to get solutions if not available in top level
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
        except AttributeError:
            # Fallback or re-raise if installation is broken
            import mediapipe.python.solutions.hands as mp_hands
            import mediapipe.python.solutions.drawing_utils as mp_drawing
            self.mp_hands = mp_hands
            self.mp_drawing = mp_drawing

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 5
        self.last_click_time = 0

    def process(self, frame):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get landmarks
                landmarks = hand_landmarks.landmark
                
                # Index finger tip (ID 8)
                index_x = int(landmarks[8].x * w)
                index_y = int(landmarks[8].y * h)
                
                # Thumb tip (ID 4)
                thumb_x = int(landmarks[4].x * w)
                thumb_y = int(landmarks[4].y * h)

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
                distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
                
                if distance < 30: # Threshold for click
                    cv2.circle(frame, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
                    if time.time() - self.last_click_time > 0.5: # Debounce
                        pyautogui.click()
                        self.last_click_time = time.time()

        return frame
