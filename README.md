# Gesture Control & Virtual Drawing

A Python application that combines virtual drawing with hand gesture control. Use your webcam to draw in the air or control your mouse cursor!

## Features

### 🎨 Drawing Mode
- **Virtual Canvas**: Draw on the screen using a colored object (default: light blue).
- **Toolbox**:
    - **Colors**: Select from Yellow, Pink, Green, or Light Blue by hovering over the top-left menu.
    - **Brush Size**: Adjust thickness by hovering over the top-right menu.
    - **Clear Screen**: Hover over the "Limpiar pantalla" button.

### ✋ Control Mode
- **Mouse Control**: Move your index finger to move the mouse cursor.
- **Click**: Pinch your thumb and index finger together to click.
- **Hand Tracking**: Uses MediaPipe for robust hand detection.

## Dependencies

- Python 3.12+
- `opencv-python`
- `numpy`
- `mediapipe`
- `pyautogui`

## Installation

1.  Clone the repository.
2.  Install the required packages:

    ```bash
    pip install opencv-python numpy mediapipe pyautogui
    ```

## Usage

Run the main script:

```bash
python lector.py
```

### Controls

- **Switch Modes**: Press `m` to toggle between **Drawing Mode** and **Control Mode**.
- **Exit**: Press `ESC`.

### Configuration

The application is set to track a "light blue" object in Drawing Mode. You can adjust the HSV values in `lector.py` (`self.celesteBajo`, `self.celesteAlto`) in the `DrawingMode` class if you want to use a different color marker.
