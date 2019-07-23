import socket   #for sockets
import sys  #for exit
import cv2
import numpy as np
#import matplotlib.pyplot as plt


#===================================================================================================================>>connecting socket
try:
	    #create an AF_INET, STREAM socket (TCP)
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error:
	print 'Failed to create socket. Error code: '
	sys.exit();

print 'Socket Created'

address='127.0.0.1'
port=3012

s.connect((address , port))


#=====================================================================================================================>>creating video object


if not cv2.useOptimized(): cv2.setUseOptimized(True)
cap=cv2.VideoCapture(1)


if not cap.isOpened(): cap.open()

print cap.get(5),cap.get(3),cap.get(4)
#=====================================================================================================================>>image processing and obtaining the cords





while(True):
	ret,frame=cap.read()
	frame0=cv2.GaussianBlur(frame,(7,7),0)
	hsv = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
	#img=frame[120:360,170:510,0]
	lower_green=np.array([80,80,20])
	upper_green=np.array([100,255,255])
	mask = cv2.inRange(hsv,lower_green,upper_green)
	#blur=cv2.GaussianBlur(mask,(3,3),0)
	blur = cv2.bilateralFilter(mask,9,75,75)
	#ret,thresh = cv2.threshold(blur,127,255,0)		#x axis is downward in the image, y axis is towards the right
	#cv2.imshow('frame1',blur)
	temp=blur.copy()


	contours,hierarchy = cv2.findContours(temp, 1, 2)
	if not contours: break
	cnt = contours[0]
	approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
	M = cv2.moments(approx)

	#rect = cv2.minAreaRect(cnt)


	(x,y),radius = cv2.minEnclosingCircle(cnt)
	center = (int(x),int(y))
	radius = int(radius)
	cv2.circle(blur,center,radius,(0,255,0),2)
	# cv2.drawContours(thresh, contours, 0, (0,255,0), 3)
	#cnt = contours[0]
	M = cv2.moments(cnt)
	print M
	if M['m00']:
		rect = cv2.minAreaRect(cnt)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(blur,[box],0,(0,0,255),2)
		print M['m00']
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		print cx,cy
		frame[cy-2:cy+2,cx-2:cx+2]=[0,0,255]
		cv2.waitKey(1) 									#==>x is to the right and y down but in the case of the matrix flip x and y directions
		cv2.imshow('frame2', frame)
	# plt.imshow(img, cmap = 'gray')
	# plt.xticks([]), plt.yticks([])
	# plt.show()



#==========================================================================================================================>>sending message


	message='[700,%d,%d]' % (-3*(cx-320),1100-2*(cy-240))      #==>>camera calibration

	try :
	    #Set the whole string
		s.sendall(message)
	except socket.error:
	    #Send failed
	    print 'Send failed'
	    sys.exit()

	print 'sent'

	reply = s.recv(1025)

	print reply
#========================================================================================>>destroy all

cap.release()
cv2.destroyAllWindows()

message0='close'
s.sendall(message0)

s.close()
