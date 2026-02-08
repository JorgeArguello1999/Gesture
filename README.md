# Virtual Gesture Drawing

A Python application that turns your webcam feed into a virtual canvas. By tracking a specific color marker (light blue by default), users can draw in the air.

## Features

- **Virtual Drawing**: Use a colored object to draw on the screen.
- **Color Selection**: Choose from 4 colors (Yellow, Pink, Green, Light Blue) by hovering over the top-left menu.
- **Brush Size Control**: Adjust line thickness by hovering over the top-right menu.
- **Clear Screen**: Quickly erase the drawing by hovering over the "Clear Screen" button.
- **Real-time Visualization**: See your drawing overlaid on the camera feed.

## Dependencies

This project requires:
- Python 3.12+
- `opencv-python`
- `numpy`

## Installation

1.  Clone the repository.
2.  Install the required packages:

    ```bash
    pip install opencv-python numpy
    ```

## Usage

Run the main script to start the application:

```bash
python lector.py
```

Ensure you have a webcam connected. The application tracks a "light blue" (celeste) color by default. You may need to adjust the HSV values in `lector.py` (`celesteBajo`, `celesteAlto`) to match your specific marker color and lighting conditions.

Press `ESC` to exit the application.
