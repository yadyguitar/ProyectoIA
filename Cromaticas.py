import cv2
import numpy as np

captura=cv2.VideoCapture(0)

def coords_canonicas(img):
	h,w,c=np.shape(img)
	for pixel in range(h*w):
		print 
	return img 


while (True):
	_,frame=captura.read()
	
	frame = cv2.flip(frame,1)
	print type(frame)
	cv2.imshow('original',frame)
	#print np.shape(frame) #w,h,c
	


	tecla=cv2.waitKey(5) & 0xFF
	if tecla == 87:
		break

cv2.destroyAllWindows()
print "finish"