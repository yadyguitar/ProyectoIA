#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''
from PIL import Image
import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock
from Entrenamiento import Entrenamiento
from getCaract import Caracteristicas

HAAR_CASCADE_PATH = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.2,
                       minDistance = 7,
                       blockSize = 10 )

class App:
    def __init__(self, video_src):
		self.track_len = 10
		self.detect_interval =2
		self.tracks = []
		self.cam = video.create_capture(video_src)
		self.frame_idx = 0
		self.out = cv2.VideoWriter('output.avi', cv2.cv.CV_FOURCC(*'DIVX'), 20.0, (640,480))
		self.distx=0
		self.disty=0
		self.x=0
		self.y=0
		self.w=0
		self.h=0

    def run(self):
    	band=0
    	i=0

        while True:
			ret, frame = self.cam.read()
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_gray=cv2.equalizeHist(frame_gray)
			vis = frame.copy()
			
			
			faces = []
			#detected = cv.HaarDetectObjects(frame, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (100,100))
			detected = cascade.detectMultiScale(frame,1.3,4,cv2.cv.CV_HAAR_SCALE_IMAGE,(20,20))
			if detected!=[]:
				#print 'face detected' #
				for (x,y,w,h) in detected: #for (x,y,w,h),n in detected:
					faces.append((x,y,w,h))
			for (self.x,self.y,self.w,self.h) in faces:
				#print 'drawing rectangle' #
					cv2.cv.Rectangle(cv2.cv.fromarray(vis), (x,y), (x+w,y+h), 255)
			
			
			if len(self.tracks) > 0:
				img0, img1 = self.prev_gray, frame_gray

				p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

				p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
				p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
				d = abs(p0-p0r).reshape(-1, 2).max(-1)
				good = d < 1
				new_tracks = []
                
				coords=[]
				for tr, (x1, y1), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
					if not good_flag:
						continue
					if x1>self.x and x1<self.x+self.w and y1>self.y and y1<self.y+self.h:
						
						tr.append((x1, y1))
						if len(tr) > self.track_len:
							del tr[0]
						new_tracks.append(tr)
						coords.append((x1,y1))
						cv2.circle(vis, (x1, y1), 2, (255, 0, 0), -1)
				try:
					temp=np.array(np.mean(coords,axis=0))
					tx,ty=temp[0],temp[1]
					cv2.circle(vis,(tx,ty),2,(0,255,0),-1)

					if faces==[]:
						print "no rostro"
						if band==0:
							self.distx=tx-self.x
							self.disty=ty-self.y
						band+=1
						self.x=np.int32(tx-self.distx)
						self.y=np.int32(ty-self.disty)
						cv2.cv.Rectangle(cv2.cv.fromarray(vis), (self.x,self.y), (self.x+self.w,self.y+self.h), 255)
					else:
						print "rostro"
						band=0

					
				except:
					print "no hay puntos de interes..."

				self.tracks = new_tracks
				#cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
				draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

			#Esto es lo primero que se ejecuta, debido a que traps en la primera es = 0, procedimiento para agreagr 
			#Detecta cada 5 frames
			if self.frame_idx % self.detect_interval == 0:
				mask = np.zeros_like(frame_gray)#crea una matriz del tamanio de la imagen de cerapios jejeje
				mask[:] = 255 #los pone todos en 255, o sea... en blancurris
				for x, y in [np.int32(tr[-1]) for tr in self.tracks]: #toma el ultimo valor de la serie de cada posicion [[(),(),(toma este)]]
					cv2.circle(mask, (x, y), 5, 0, -1)
				p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

				if p is not None:
					for x, y in np.float32(p).reshape(-1, 2):
						self.tracks.append([(x, y)])


			self.frame_idx += 1
			self.prev_gray = frame_gray
			cv2.imshow('lk_track', vis)
			self.out.write(vis)
			i+=1
			ch = 0xFF & cv2.waitKey(1)
			
			if ch==113:
				try:
					crop_img = frame[self.y:self.y+self.h, self.x:self.x+self.w] # Crop from x, y, w, h -> 100, 200, 300, 400
					# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
					crop_img=cv2.resize(crop_img,(80,80))
					cv2.imwrite("a.jpg",crop_img)
					Entrenamiento().clasificar(Caracteristicas(5).getCaract("a.jpg"))
				except:
					print "error crop"
			if ch == 27:
				break

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = 0

    print __doc__

    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
