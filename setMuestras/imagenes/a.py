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

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock


HAAR_CASCADE_PATH = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
		self.track_len = 10
		self.detect_interval =2
		self.tracks = []
		self.cam = video.create_capture(video_src)
		self.frame_idx = 0
		self.out = cv2.VideoWriter('output.avi', cv2.cv.CV_FOURCC(*'DIVX'), 20.0, (640,480))

    def run(self):
    	i=0
        while True:
			ret, frame = self.cam.read()
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			vis = frame.copy()
			
			x=0
			y=0
			w=0
			y=0
			faces = []
			#detected = cv.HaarDetectObjects(frame, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (100,100))
			detected = cascade.detectMultiScale(frame,1.3,4,cv2.cv.CV_HAAR_SCALE_IMAGE,(20,20))
			if detected!=[]:
				#print 'face detected' #
				for (x,y,w,h) in detected: #for (x,y,w,h),n in detected:
					faces.append((x,y,w,h))
			for (x,y,w,h) in faces:
				#print 'drawing rectangle' #
					cv2.cv.Rectangle(cv2.cv.fromarray(vis), (x,y), (x+w,y+h), 255)
			
			
			if len(self.tracks) > 0:
				for element in self.tracks:
					if element[0][0]<x or element[0][0]>x+w or element[0][1]<y or element[0][1]>y+h:
						pass#self.tracks.remove(element)

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
					if x1>x and x1<x+w and y1>y and y1<y+h:
						print x,y,x+w,y+h," y ",x1,y1
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
