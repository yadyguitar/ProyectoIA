import cv2
import numpy as np


img= cv2.imread('00.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.equalizeHist(img)
#img = np.hstack((img,res)) #stacking images side-by-side
cv2.imshow("0",img)


img1= cv2.imread('01.jpg')

img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1 = cv2.equalizeHist(img1)
#img1 = np.hstack((img1,res1)) #stacking images side-by-side
cv2.imshow("1",img1)

key=cv2.waitKey(0) & 0xFF

if key == ord('e'):
	cv2.destroyAllWindows()