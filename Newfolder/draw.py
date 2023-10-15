import cv2 as cv
import numpy as np
img=cv.imread('New folder/th.jpeg')
blank=np.zeros((500,500),dtype='unit8')
cv.imshow("Blank",blank)
cv.imshow('cat',img)
cv.waitKey(0)
cv.destroyAllWindows()