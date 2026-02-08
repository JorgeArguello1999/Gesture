import cv2
import sys
import os

# Add the current directory to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.drawing import DrawingMode
from modules.hand_control import HandControlMode

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    drawing_mode = DrawingMode()
    try:
        hand_control_mode = HandControlMode()
    except Exception as e:
        print(f"Error initializing hand control: {e}")
        hand_control_mode = None
    
    current_mode = "DRAWING" # drawing or control

    print("Application Started")
    print("Press 'm' to switch modes")
    print("Press 'ESC' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)

        # Show current mode
        cv2.putText(frame, f"Mode: {current_mode}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if current_mode == "DRAWING":
            frame = drawing_mode.process(frame)
        elif current_mode == "CONTROL":
            if hand_control_mode:
                try:
                    frame = hand_control_mode.process(frame)
                except Exception as e:
                    print(f"Runtime error in hand control: {e}")
            else:
                 cv2.putText(frame, "Hand Control Disabled (Error)", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Gesture App', frame)

        k = cv2.waitKey(1)
        if k == 27: # ESC
            break
        elif k == ord('m'):
            if current_mode == "DRAWING":
                current_mode = "CONTROL"
            else:
                current_mode = "DRAWING"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
