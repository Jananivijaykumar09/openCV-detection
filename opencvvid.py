import cv2
import numpy as np
capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
while True:
    ret,frame = capture.read()
    width = int(capture.get(3))
    height = int(capture.get(4))
    image = np.zeros(frame.shape,np.uint8)
    smaller_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    image[:height//2,:width//2] = cv2.rotate(smaller_frame,cv2.cv2.ROTATE_180)
    image[height//2:,:width//2] = smaller_frame
    image[:height//2,width//2:] = cv2.rotate(smaller_frame,cv2.cv2.ROTATE_180)
    image[height//2:,width//2:] = smaller_frame
    

    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([7,59,7])
    upper_blue = np.array([112,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)
    result = cv2.bitwise_and(image,image,mask=mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 5, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 87, 155), 5)

    cv2.imshow('frame', frame)
    cv2.imshow('mask',mask)
    cv2.imshow('frames',result)
    cv2.imshow('cross',frame)
    cv2.imshow('tiles',image)
    if cv2.waitKey(1)==ord('q'):
       break
capture.release()
cv2.destroyAllWindows
