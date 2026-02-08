import mediapipe as mp
import sys

print(f"Mediapipe version: {mp.__version__}")

try:
    import mediapipe.tasks.python.vision.hand_landmarker
    print("Tasks API HandLandmarker available")
except ImportError as e:
    print(f"Tasks API HandLandmarker missing: {e}")
except AttributeError as e:
    print(f"Tasks API HandLandmarker attribute missing: {e}")

try:
    print(mp.solutions.hands)
    print("Solutions API available")
except AttributeError:
    print("Solutions API missing")
