import cv2
import sys
import os

# Add the current directory to sys.path to ensure modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.drawing import DrawingMode
from modules.hand_control import HandControlMode

def main():
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
    
    # App State
    current_mode = "MENU" # MENU, DRAWING, CONTROL
    
    # Mouse Callback for Menu
    def menu_callback(event, x, y, flags, param):
        nonlocal current_mode
        # We need w and h to calculate button positions, but they are local to the loop.
        # However, we can use the last known frame size or cap.get()
        w = int(cap.get(3))
        h = int(cap.get(4))
        
        # Calculate dynamic button positions
        # Button 1: Drawing Mode (Left side - 10% to 40% width)
        x1_start, x1_end = int(w * 0.1), int(w * 0.4)
        y_start, y_end = int(h * 0.3), int(h * 0.6)
        
        # Button 2: Control Mode (Right side - 60% to 90% width)
        x2_start, x2_end = int(w * 0.6), int(w * 0.9)
        
        # Button 3: Exit (Bottom Center - 40% to 60% width)
        x3_start, x3_end = int(w * 0.4), int(w * 0.6)
        y3_start, y3_end = int(h * 0.7), int(h * 0.85)

        if current_mode == "MENU" and event == cv2.EVENT_LBUTTONDOWN:
            if x1_start < x < x1_end and y_start < y < y_end:
                current_mode = "DRAWING"
            elif x2_start < x < x2_end and y_start < y < y_end:
                current_mode = "CONTROL"
            elif x3_start < x < x3_end and y3_start < y < y3_end:
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
            y_start, y_end = int(h * 0.3), int(h * 0.6)
            x2_start, x2_end = int(w * 0.6), int(w * 0.9)
            x3_start, x3_end = int(w * 0.4), int(w * 0.6)
            y3_start, y3_end = int(h * 0.7), int(h * 0.85)

            # Draw Menu
            cv2.rectangle(frame, (0, 0), (w, h), (20, 20, 20), -1)
            
            # Title
            title = "Gesture Control App"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.putText(frame, title, (w//2 - title_size[0]//2, int(h * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Button 1: Drawing Mode
            cv2.rectangle(frame, (x1_start, y_start), (x1_end, y_end), (89, 222, 255), -1)
            btn1_text = "Drawing Mode"
            btn1_size = cv2.getTextSize(btn1_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(frame, btn1_text, (x1_start + (x1_end-x1_start)//2 - btn1_size[0]//2, y_start + (y_end-y_start)//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Button 2: Control Mode
            cv2.rectangle(frame, (x2_start, y_start), (x2_end, y_end), (128, 0, 255), -1)
            btn2_text = "Control Mode"
            btn2_size = cv2.getTextSize(btn2_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(frame, btn2_text, (x2_start + (x2_end-x2_start)//2 - btn2_size[0]//2, y_start + (y_end-y_start)//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Button 3: Exit
            cv2.rectangle(frame, (x3_start, y3_start), (x3_end, y3_end), (0, 0, 255), -1)
            btn3_text = "Exit"
            btn3_size = cv2.getTextSize(btn3_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(frame, btn3_text, (x3_start + (x3_end-x3_start)//2 - btn3_size[0]//2, y3_start + (y3_end-y3_start)//2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Instructions
            instr = "Select a mode to start"
            instr_size = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.putText(frame, instr, (w//2 - instr_size[0]//2, int(h * 0.95)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

        if current_mode == "DRAWING":
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
            # Show current mode
            cv2.putText(frame, f"Mode: {current_mode}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

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
