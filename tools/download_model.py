import urllib.request

url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
urllib.request.urlretrieve(url, "hand_landmarker.task")
print("Model downloaded to hand_landmarker.task")
