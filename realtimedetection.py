import cv2 # it is the popular open source library for CV , ML & image processing
#It allows you to work with images and videos in real time,
from keras.models import model_from_json
import numpy as np

# Load model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # for our camera acces
face_cascade = cv2.CascadeClassifier(haar_file)

# Feature extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img = extract_features(roi_gray)
        pred = model.predict(img)
        label = labels[pred.argmax()]
        cv2.putText(frame, label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    cv2.imshow("Facial Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

webcam.release()
cv2.destroyAllWindows()
