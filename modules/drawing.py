import cv2
import numpy as np

class DrawingMode:
    def __init__(self):
        # HSV Colors for Detection (Blue marker default)
        self.celesteBajo = np.array([75, 185, 88], np.uint8)
        self.celesteAlto = np.array([112, 255, 255], np.uint8)

        # Drawing Colors (BGR)
        self.colors = {
            "Blue": (255, 113, 82),   # #5271FF
            "Yellow": (89, 222, 255), # #FFDE59
            "Pink": (128, 0, 255),    # #FF0080
            "Green": (0, 255, 36),    # #24FF00
            "Eraser": (0, 0, 0)       # Eraser (Black on mask)
        }
        
        # UI Configuration
        self.header_height = 80
        self.button_radius = 25
        self.buttons = [
            {"name": "Blue", "color": self.colors["Blue"], "center": (100, 40)},
            {"name": "Yellow", "color": self.colors["Yellow"], "center": (180, 40)},
            {"name": "Pink", "color": self.colors["Pink"], "center": (260, 40)},
            {"name": "Green", "color": self.colors["Green"], "center": (340, 40)},
            {"name": "Eraser", "color": (200, 200, 200), "center": (450, 40)}, # Visual color for button
            {"name": "Clear", "color": (50, 50, 200), "center": (1200, 40)} # Far right
        ]

        # State
        self.current_color = self.colors["Blue"]
        self.brush_thickness = 6
        self.eraser_thickness = 50
        self.x1 = None
        self.y1 = None
        self.imAux = None
        self.needs_clear = False
        
        # Calibration State
        self.click_point = None

    def handle_click(self, x, y):
        """Receives click coordinates from main loop for calibration"""
        # If clicked in header, ignore (handled by UI logic typically, but UI logic depends on detection)
        # Actually, let's allow clicking ANYWHERE to calibrate if we want, 
        # BUT we have UI buttons at the top. 
        # Let's say: Click on top bar = UI Action. Click on canvas = Calibration (if enabled)
        # For simplicity, let's just assume all clicks NOT on buttons are calibration attempts?
        # Or better: Add a "Calibrate" button to the UI to toggle a mode?
        # User request: "quiero que pueda seguir otro color".
        # Simplest UX: Just click on the object you want to track.
        # But we need to avoid accidental recalibration when drawing.
        # Let's add a "Pick Color" button to the UI.
        self.click_point = (x, y)

    def process(self, frame):
        h, w, c = frame.shape
        
        # Initialize or Clear Canvas safely
        if self.imAux is None or self.needs_clear:
            self.imAux = np.zeros(frame.shape, dtype=np.uint8)
            self.needs_clear = False
            
        # Update Clear Button Position based on current frame width
        self.buttons[-1]["center"] = (w - 80, 40)

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Handle Calibration Click
        if self.click_point:
            cx, cy = self.click_point
            # Check if inside UI header
            if cy < self.header_height:
                 pass # UI click, handled in loop logic or via check_ui_interaction?
                 # Wait, interaction logic was reliant on detecting the marker moving over buttons.
                 # Mouse click is different. 
                 # Let's just use the mouse click for calibration for now to keep it simple as requested.
            else:
                # Sample Color
                pixel = frameHSV[cy, cx]
                h_val, s_val, v_val = pixel
                
                # Define new range with some tolerance
                # Tolerance: H +/- 10, S +/- 40, V +/- 40
                self.celesteBajo = np.array([max(0, h_val - 10), max(50, s_val - 40), max(50, v_val - 40)], np.uint8)
                self.celesteAlto = np.array([min(179, h_val + 10), 255, 255], np.uint8)
                
                print(f"Calibrated to: HSV[{h_val}, {s_val}, {v_val}]")
            
            self.click_point = None

        # Detect Marker
        mask = cv2.inRange(frameHSV, self.celesteBajo, self.celesteAlto)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.medianBlur(mask, 13)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        # Draw UI Overlay
        cv2.rectangle(frame, (0, 0), (w, self.header_height), (50, 50, 50), -1) # Header BG
        cv2.rectangle(frame, (0, 0), (w, self.header_height), (100, 100, 100), 2) # Border
        
        cv2.putText(frame, "CLICK object to track it!", (w//2 - 100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for btn in self.buttons:
            # Draw Button
            cv2.circle(frame, btn["center"], self.button_radius, btn["color"], -1)
            cv2.circle(frame, btn["center"], self.button_radius, (200, 200, 200), 2) # Ring
            
            # Label
            if btn["name"] == "Eraser":
                 cv2.putText(frame, "Eraser", (btn["center"][0]-25, btn["center"][1]+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            elif btn["name"] == "Clear":
                 cv2.putText(frame, "Clear All", (btn["center"][0]-30, btn["center"][1]+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # Highlight Selected
            is_selected = False
            if btn["name"] == "Eraser" and self.current_color == self.colors["Eraser"]:
                is_selected = True
            elif btn["name"] in self.colors and self.colors[btn["name"]] == self.current_color:
                is_selected = True
            
            if is_selected:
                 cv2.circle(frame, btn["center"], self.button_radius + 5, (0, 255, 0), 3)


        for c in cnts:
            area = cv2.contourArea(c)
            if area > 1000:
                x, y, w_box, h_box = cv2.boundingRect(c)
                x2 = x + w_box // 2
                y2 = y
                
                # Check UI Interaction (Pointer in Header)
                if y2 < self.header_height:
                    self._check_ui_interaction(x2, y2)
                    self.x1, self.y1 = None, None # Don't draw while selecting
                else:
                    # Draw on Canvas
                    if self.x1 is not None:
                        # Draw line
                        thickness = self.eraser_thickness if self.current_color == self.colors["Eraser"] else self.brush_thickness
                        
                        cv2.line(self.imAux, (self.x1, self.y1), (x2, y2), self.current_color, thickness)
                    
                    self.x1 = x2
                    self.y1 = y2
                
                # Visual Feedback of Pointer
                cv2.circle(frame, (x2, y2), 5, self.current_color, -1)
            else:
                self.x1, self.y1 = None, None

        # Merge Canvas with Frame
        imAuxGray = cv2.cvtColor(self.imAux, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(imAuxGray, 10, 255, cv2.THRESH_BINARY)
        thInv = cv2.bitwise_not(th)
        frame = cv2.bitwise_and(frame, frame, mask=thInv)
        frame = cv2.add(frame, self.imAux)

        return frame

    def _check_ui_interaction(self, x, y):
        # Specific hit testing for buttons
        for btn in self.buttons:
            bx, by = btn["center"]
            if np.hypot(x - bx, y - by) < self.button_radius:
                if btn["name"] == "Clear":
                    self.needs_clear = True
                elif btn["name"] == "Eraser":
                    self.current_color = self.colors["Eraser"]
                elif btn["name"] in self.colors:
                    self.current_color = self.colors[btn["name"]]
