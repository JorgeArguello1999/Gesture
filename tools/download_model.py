import urllib.request

url_hand = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
urllib.request.urlretrieve(url_hand, "hand_landmarker.task")
print("Model downloaded to hand_landmarker.task")

url_face = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
urllib.request.urlretrieve(url_face, "face_landmarker.task")
print("Model downloaded to face_landmarker.task")
