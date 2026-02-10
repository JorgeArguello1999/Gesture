import mediapipe as mp
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
