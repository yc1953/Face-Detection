import cv2

# Including Haarcascade which contains informations about many faces already detected.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Switching on the camera to detect video.
cap = cv2.VideoCapture(0)

# When camera will be opened
while cap.isOpened():
    _, img = cap.read()
    
    # Changing the frame to grayscale    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Drawing the rectangles at the co-ordinates returned from the 'detectMultiScale' method.    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
    # Showing the image.
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
