import cv2
import numpy as np

class DrawingMode:
    def __init__(self):
        # HSV Colors
        self.celesteBajo = np.array([75, 185, 88], np.uint8)
        self.celesteAlto = np.array([112, 255, 255], np.uint8)

        # Drawing Colors
        self.colorCeleste = (255, 113, 82)
        self.colorAmarillo = (89, 222, 255)
        self.colorRosa = (128, 0, 255)
        self.colorVerde = (0, 255, 36)
        self.colorLimpiarPantalla = (29, 112, 246)

        # Thickness
        self.grosorCeleste = 6
        self.grosorAmarillo = 2
        self.grosorRosa = 2
        self.grosorVerde = 2
        
        self.grosorPeque = 6
        self.grosorMedio = 1
        self.grosorGrande = 1

        # State
        self.color = self.colorCeleste
        self.grosor = 3
        self.x1 = None
        self.y1 = None
        self.imAux = None

    def process(self, frame):
        # Initialize imAux if needed
        if self.imAux is None:
            self.imAux = np.zeros(frame.shape, dtype=np.uint8)

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Draw UI
        self._draw_ui(frame)

        # Detect Color
        maskCeleste = cv2.inRange(frameHSV, self.celesteBajo, self.celesteAlto)
        maskCeleste = cv2.erode(maskCeleste, None, iterations=1)
        maskCeleste = cv2.dilate(maskCeleste, None, iterations=2)
        maskCeleste = cv2.medianBlur(maskCeleste, 13)
        
        cnts, _ = cv2.findContours(maskCeleste, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 1000:
                x, y2, w, h = cv2.boundingRect(c)
                x2 = x + w // 2

                if self.x1 is not None:
                    # Check UI interactions
                    if 0 < y2 < 50:
                        self._check_ui_interaction(x2)
                    
                    # Draw
                    if 0 < y2 < 60 or 0 < self.y1 < 60:
                        pass # Don't draw in UI area
                    else:
                        self.imAux = cv2.line(self.imAux, (self.x1, self.y1), (x2, y2), self.color, self.grosor)
                
                cv2.circle(frame, (x2, y2), self.grosor, self.color, 3)
                self.x1 = x2
                self.y1 = y2
            else:
                self.x1, self.y1 = None, None

        # Merge layers
        imAuxGray = cv2.cvtColor(self.imAux, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(imAuxGray, 10, 255, cv2.THRESH_BINARY)
        thInv = cv2.bitwise_not(th)
        frame = cv2.bitwise_and(frame, frame, mask=thInv)
        frame = cv2.add(frame, self.imAux)

        return frame

    def _draw_ui(self, frame):
        # Colors
        cv2.rectangle(frame, (0, 0), (50, 50), self.colorAmarillo, self.grosorAmarillo)
        cv2.rectangle(frame, (50, 0), (100, 50), self.colorRosa, self.grosorRosa)
        cv2.rectangle(frame, (100, 0), (150, 50), self.colorVerde, self.grosorVerde)
        cv2.rectangle(frame, (150, 0), (200, 50), self.colorCeleste, self.grosorCeleste)

        # Clear Screen
        cv2.rectangle(frame, (300, 0), (400, 50), self.colorLimpiarPantalla, 1)
        cv2.putText(frame, 'Limpiar', (320, 20), 6, 0.6, self.colorLimpiarPantalla, 1, cv2.LINE_AA)
        cv2.putText(frame, 'pantalla', (320, 40), 6, 0.6, self.colorLimpiarPantalla, 1, cv2.LINE_AA)

        # Thickness
        cv2.rectangle(frame, (490, 0), (540, 50), (0, 0, 0), self.grosorPeque)
        cv2.circle(frame, (515, 25), 3, (0, 0, 0), -1)
        cv2.rectangle(frame, (540, 0), (590, 50), (0, 0, 0), self.grosorMedio)
        cv2.circle(frame, (565, 25), 7, (0, 0, 0), -1)
        cv2.rectangle(frame, (590, 0), (640, 50), (0, 0, 0), self.grosorGrande)
        cv2.circle(frame, (615, 25), 11, (0, 0, 0), -1)

    def _check_ui_interaction(self, x2):
        # Color Selection
        if 0 < x2 < 50:
            self.color = self.colorAmarillo
            self.grosorAmarillo = 6; self.grosorRosa = 2; self.grosorVerde = 2; self.grosorCeleste = 2
        elif 50 < x2 < 100:
            self.color = self.colorRosa
            self.grosorAmarillo = 2; self.grosorRosa = 6; self.grosorVerde = 2; self.grosorCeleste = 2
        elif 100 < x2 < 150:
            self.color = self.colorVerde
            self.grosorAmarillo = 2; self.grosorRosa = 2; self.grosorVerde = 6; self.grosorCeleste = 2
        elif 150 < x2 < 200:
            self.color = self.colorCeleste
            self.grosorAmarillo = 2; self.grosorRosa = 2; self.grosorVerde = 2; self.grosorCeleste = 6
        
        # Clear Screen
        elif 300 < x2 < 400:
            self.imAux = None # Will be re-initialized in process()

        # Thickness
        elif 490 < x2 < 540:
            self.grosor = 3
            self.grosorPeque = 6; self.grosorMedio = 1; self.grosorGrande = 1
        elif 540 < x2 < 590:
            self.grosor = 7
            self.grosorPeque = 1; self.grosorMedio = 6; self.grosorGrande = 1
        elif 590 < x2 < 640:
            self.grosor = 11
            self.grosorPeque = 1; self.grosorMedio = 1; self.grosorGrande = 6
