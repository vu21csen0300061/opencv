import cv2 as cv
import numpy as np
img=cv.imread('Newfolder/recg.jpg')
cv.imshow('people',img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

haar_classifier=cv.CascadeClassifier('haar.xml')
face_detect=haar_classifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h) in face_detect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('detected',img)

cv.waitKey(0)
cv.destroyAllWindows()