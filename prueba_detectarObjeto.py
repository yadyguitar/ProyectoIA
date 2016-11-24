import cv2
import cv


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(1)


out=cv2.VideoWriter('output.avi',cv.CV_FOURCC('D','I','V','X'), 25, (640,480))



while (True):
	ret, img = cap.read()

	cv2.line(img,(215,0),(215,480),(255,0,0),2)
	cv2.line(img,(430,0),(430,480),(255,0,0),2)

	cv2.line(img,(0,160),(645,160),(255,0,0),2)
	cv2.line(img,(0,320),(645,320),(255,0,0),2)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print faces
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	out.write(img)
	cv2.imshow('frame',img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

