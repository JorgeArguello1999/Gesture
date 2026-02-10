import mediapipe as mp
import sys
import os

# Add parent directory to path to allow importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(f"MP dir: {dir(mp)}")
try:
    import mediapipe.solutions
    print("Imported mediapipe.solutions")
    print(f"Solutions dir: {dir(mediapipe.solutions)}")
except ImportError as e:
    print(f"Failed to import mediapipe.solutions: {e}")

try:
    import mediapipe.python.solutions
    print("Imported mediapipe.python.solutions")
except ImportError as e:
    print(f"Failed to import mediapipe.python.solutions: {e}")
