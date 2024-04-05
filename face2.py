import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifiers for face detection
face_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')  # Corrected name

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the first face detector
    faces1 = face_cascade1.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Detect faces using the second face detector
    faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Check if either faces1 or faces2 is not empty before concatenating
    if faces1 is not None and len(faces1) > 0:
        faces = faces1
    elif faces2 is not None and len(faces2) > 0:
        faces = faces2
    else:
        faces = np.empty((0, 4), dtype=np.int32)  # Empty array if both are empty

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the number of persons detected
    num_persons = len(faces)
    cv2.putText(frame, f'Persons: {num_persons}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with rectangles around faces
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
