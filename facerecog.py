import numpy as np
import cv2 as cv
import os
people=['jk','mahesh','ronaldo','v']
dir=r'C:\Users\Roshini\pics\train'
har=cv.CascadeClassifier('haar.xml')
features=[]
labels=[]
def cratetrain():
    for person in people:
        path=os.path.join(dir,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            face_detect=har.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in face_detect:
                face_rec=gray[y:y+h,x:x+w]
                features.append(face_rec)
                labels.append(label)
cratetrain()
features=np.array(features,dtype='object')
labels=np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
np.save('features.npy',features)
np.save('labels.npy',labels)

test=cv.imread('C:/Users/Roshini/pics/train/jk/th.jpeg')
grey=cv.cvtColor(test,cv.COLOR_BGR2GRAY)
face=har.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=4)
for (x,y,w,h) in face:
    faces=grey[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(faces)
    cv.putText(test,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(test,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('detected image',test)
cv.waitKey(0)