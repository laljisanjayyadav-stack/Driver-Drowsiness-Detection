import cv2
import pygame
import os
import urllib.request
import winsound

# sound setup
pygame.mixer.init()

# download cascades if not present
if not os.path.exists('haarcascade_frontalface_default.xml'):
    print("downloading face cascade...")
    url1 = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url1, 'haarcascade_frontalface_default.xml')

if not os.path.exists('haarcascade_eye.xml'):
    print("downloading eye cascade...")
    url2 = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    urllib.request.urlretrieve(url2, 'haarcascade_eye.xml')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cam = cv2.VideoCapture(0)

c = 0

while True:
    ret, img = cam.read()
    if ret == False:
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    msg = "Status: Normal"
    color = (0, 255, 0)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            roi_gray = gray[y : y + int(h/1.8), x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 8)

            if len(eyes) == 0:
                c = c + 1
                if c >= 10:
                    msg = "ALERT! SLEEPING DETECTED"
                    color = (0, 0, 255)
                    # play alarm sound
                    winsound.Beep(2000, 400)
            else:
                c = 0
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 1)
    else:
        c = 0

    cv2.putText(img, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Drowsiness Test", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()