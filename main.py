import cv2
import sys
import os
import subprocess

# Add the current directory to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.drawing import DrawingMode
from modules.hand_control import HandControlMode
from modules.eye_control import EyeControlMode

def check_and_download_models():
    """Checks if models exist, if not, runs the download script."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    hand_model = os.path.join(models_dir, "hand_landmarker.task")
    face_model = os.path.join(models_dir, "face_landmarker.task")
    
    missing = False
    if not os.path.exists(hand_model):
        print(f"Missing model: {hand_model}")
        missing = True
    if not os.path.exists(face_model):
        print(f"Missing model: {face_model}")
        missing = True
        
    if missing:
        print("Models are missing. Attempting to download...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "download_model.py")
        try:
            subprocess.run([sys.executable, script_path], check=True)
            print("Models downloaded successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download models: {e}")
            sys.exit(1)
        except Exception as e:
             print(f"An error occurred while downloading models: {e}")
             sys.exit(1)
    else:
        print("All models verified.")

def main():
    check_and_download_models()

    cap = cv2.VideoCapture(0)
    
    # Increase resolution for better UI
    cap.set(3, 1280)
    cap.set(4, 720)

    drawing_mode = DrawingMode()
    
    try:
        hand_control_mode = HandControlMode()
    except Exception as e:
        print(f"Error initializing hand control: {e}")
        hand_control_mode = None
        
    try:
        eye_control_mode = EyeControlMode()
    except Exception as e:
        print(f"Error initializing eye control: {e}")
        eye_control_mode = None
    
    # App State
    current_mode = "MENU" # MENU, DRAWING, CONTROL, EYE_CONTROL
    
    # Mouse Callback for Menu
    def menu_callback(event, x, y, flags, param):
        nonlocal current_mode
        w = int(cap.get(3))
        h = int(cap.get(4))
        
        # Calculate dynamic button positions
        # Row 1
        # Button 1: Drawing Mode (Left)
        x1_start, x1_end = int(w * 0.1), int(w * 0.4)
        y1_start, y1_end = int(h * 0.2), int(h * 0.4)
        
        # Button 2: Hand Control (Right)
        x2_start, x2_end = int(w * 0.6), int(w * 0.9)
        y2_start, y2_end = int(h * 0.2), int(h * 0.4)
        
        # Row 2
        # Button 3: Eye Control (Center)
        x3_start, x3_end = int(w * 0.35), int(w * 0.65)
        y3_start, y3_end = int(h * 0.5), int(h * 0.7)
        
        # Row 3
        # Button 4: Exit (Bottom)
        x4_start, x4_end = int(w * 0.4), int(w * 0.6)
        y4_start, y4_end = int(h * 0.8), int(h * 0.9)

        if current_mode == "MENU" and event == cv2.EVENT_LBUTTONDOWN:
            if x1_start < x < x1_end and y1_start < y < y1_end:
                current_mode = "DRAWING"
            elif x2_start < x < x2_end and y2_start < y < y2_end:
                current_mode = "CONTROL"
            elif x3_start < x < x3_end and y3_start < y < y3_end:
                current_mode = "EYE_CONTROL"
            elif x4_start < x < x4_end and y4_start < y < y4_end:
                sys.exit()

    cv2.namedWindow('Gesture App')
    cv2.setMouseCallback('Gesture App', menu_callback)

    print("Application Started")
    print("Press 'ESC' to exit")
    print("Press 'm' to return to menu")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if current_mode == "MENU":
            # Dynamic Layout
            x1_start, x1_end = int(w * 0.1), int(w * 0.4)
            y1_start, y1_end = int(h * 0.2), int(h * 0.4)
            
            x2_start, x2_end = int(w * 0.6), int(w * 0.9)
            y2_start, y2_end = int(h * 0.2), int(h * 0.4)
            
            x3_start, x3_end = int(w * 0.35), int(w * 0.65)
            y3_start, y3_end = int(h * 0.5), int(h * 0.7)
            
            x4_start, x4_end = int(w * 0.4), int(w * 0.6)
            y4_start, y4_end = int(h * 0.8), int(h * 0.9)

            # Draw Menu Background
            cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 20), -1)
            
            # Title
            title = "Gesture Control App"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.putText(frame, title, (w//2 - title_size[0]//2, int(h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Button 1: Drawing Mode
            cv2.rectangle(frame, (x1_start, y1_start), (x1_end, y1_end), (89, 222, 255), -1)
            cv2.putText(frame, "Drawing Mode", (x1_start + 20, y1_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Button 2: Hand Control
            cv2.rectangle(frame, (x2_start, y2_start), (x2_end, y2_end), (128, 0, 255), -1)
            cv2.putText(frame, "Hand Control", (x2_start + 20, y2_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Button 3: Eye Control
            cv2.rectangle(frame, (x3_start, y3_start), (x3_end, y3_end), (0, 255, 0), -1)
            cv2.putText(frame, "Eye Control", (x3_start + 40, y3_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            # Button 4: Exit
            cv2.rectangle(frame, (x4_start, y4_start), (x4_end, y4_end), (0, 0, 255), -1)
            cv2.putText(frame, "Exit", (x4_start + 60, y4_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif current_mode == "DRAWING":
            frame = drawing_mode.process(frame)
            cv2.putText(frame, "Press 'm' for Menu", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        elif current_mode == "CONTROL":
            if hand_control_mode:
                try:
                    frame = hand_control_mode.process(frame)
                except Exception as e:
                    print(f"Runtime error in hand control: {e}")
            else:
                 cv2.putText(frame, "Hand Control Disabled (Error)", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'm' for Menu", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        elif current_mode == "EYE_CONTROL":
            if eye_control_mode:
                try:
                    frame = eye_control_mode.process(frame)
                except Exception as e:
                    print(f"Runtime error in eye control: {e}")
            else:
                 cv2.putText(frame, "Eye Control Disabled (Error)", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, "Press 'm' for Menu", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Gesture App', frame)

        k = cv2.waitKey(1)
        if k == 27: # ESC
            break
        elif k == ord('m'):
            current_mode = "MENU"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
