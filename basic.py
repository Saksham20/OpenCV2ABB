import socket   #for sockets
import sys  #for exit
import cv2
import numpy as np
import matplotlib.pyplot as plt






# if not cv2.useOptimized(): cv2.setUseOptimized(True)
# cap=cv2.VideoCapture(0)

# if not cap.isOpened(): cap.open()

# print cap.get(5),cap.get(3),cap.get(4)

green0 = np.uint8([[[135,170,105 ]]])
green1 = np.uint8([[[145,175,115 ]]])
hsv_green0 = cv2.cvtColor(green0,cv2.COLOR_BGR2HSV)
hsv_green1 = cv2.cvtColor(green1,cv2.COLOR_BGR2HSV)
print hsv_green0,hsv_green1
while(True):
ret,frame=cap.read()
img=cv2.imread(frame,1)
cv2.imwrite('test.jpg',img)
	img=frame[10,630]
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	print hsv
	img=frame[120:360,170:510,0]

	ret,thresh = cv2.threshold(img,50,255,0)
	cv2.imshow('frame2',thresh)
	#cv2.imshow('frame3',thresh)
	contours,hierarchy = cv2.findContours(thresh, 1, 2)
	cnt0 = contours[4]
	cv2.drawContours(thresh, contours, 0, (0,255,0), 3)
	cnt = contours[0]
	M = cv2.moments(cnt)
	print M
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	print cx,cy
	fgmask = fgbg.apply(frame)

	img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	ret,th1 = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV)
	r,co=th1.shape
	r1,c1=img.shape

	x=0
	y=0
	c=0
	th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	for a in range(r):			#iteration over all the pixel values
		for b in range(co):
			val=th1[r-1][co-1]
			c=c+val
			x=x+ ((b+1)*val)
			y=y+ ((a+1)*val)

# ======================================================================================

	x=x/c
	y=y/c

	X=700+(x)  		#transformation from the image space to the world space
	Y=700+(y)
	print X,Y


	gray = cv2.flip(gray,1)
	plt.imshow(frame)
	plt.show()
	all the image processing comes here...it acts on a specific frame

	cv2.imshow('frame',thresh)



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break




cap.release()
cv2.destroyAllWindows()
