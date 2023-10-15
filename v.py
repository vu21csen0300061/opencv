import cv2 as cv
import numpy as np
img=cv.imread('Newfolder/Cat.jpeg')
#blank=np.zeros((500,500,3),dtype='uint8')
#blank[:]=0,255,0
#cv.imshow("Blank",blank)
#grey=cv.cvtColor(img,cv.COLOR_BAYER_BG2GRAY)
#cv.imshow("grey",grey)

#blur=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)
#cv.imshow("blur",blur)

canny=cv.Canny(img,155,175)
cv.imshow("canny",canny)

dil=cv.dilate(canny,(3,3),iterations=1)
cv.imshow("dilate",dil)

erode=cv.erode(dil,(3,3),iterations=1)
cv.imshow("erode",erode)

#cv.rectangle(blank,(0,0),(250,250),(0,255,0),thickness=2)
#cv.imshow("rectangle",blank)

#cv.rectangle(blank,(0,0),(250,250),(0,255,0),thickness=cv.FILLED)
#cv.imshow("rectangle",blank)

#cv.circle(blank,(250,250),40,(255,0,0),thickness=-1)
#cv.imshow("circle",blank)

#cv.line(blank,(0,0),(250,250),(255,255,255),thickness=2)
#cv.imshow("line",blank)

#cv.putText(blank,'hello',(250,250),cv.FONT_HERSHEY_COMPLEX,1.0,(255,0,0),thickness=2)
#cv.imshow("text",blank)


cv.waitKey(0)
cv.destroyAllWindows()