import cv2
import numpy as np
import matplotlib.pyplot as plt


if not cv2.useOptimized(): cv2.setUseOptimized(True)
cap=cv2.VideoCapture(1)


if not cap.isOpened(): cap.open()

print cap.get(5),cap.get(3),cap.get(4)
#fgbg = cv2.BackgroundSubtractorMOG()
lower_green=np.array([80,80,20])
upper_green=np.array([100,255,255])	
# lower_green=np.array([100,100,100])
# upper_green=np.array([115,255,255])


while(True):
	ret,frame=cap.read()
	frame0=cv2.GaussianBlur(frame,(7,7),0)
	hsv = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
	#img=frame[120:360,170:510,0]	
	
	mask = cv2.inRange(hsv,lower_green,upper_green)
	#blur=cv2.GaussianBlur(mask,(3,3),0)
	blur = cv2.bilateralFilter(mask,9,75,75)
	#ret,thresh = cv2.threshold(blur,127,255,0)		#x axis is downward in the image, y axis is towards the right
	cv2.imshow('frame1',blur)
	temp=blur.copy()
	

	contours,hierarchy = cv2.findContours(temp, 1, 2)
	if not contours: break
	cnt = contours[0]
	approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
	hull = cv2.convexHull(cnt)
	M = cv2.moments(hull)
	
	
	

	#(x,y),radius = cv2.minEnclosingCircle(hull)
	# center = (int(x),int(y))
	# radius = int(radius)
	# cv2.circle(blur,center,radius,(0,255,0),2)
	# # cv2.drawContours(thresh, contours, 0, (0,255,0), 3)
	# #cnt = contours[0]
	#M = cv2.moments(cnt)
	# print M
	if M['m00']: 
		rect = cv2.minAreaRect(cnt)
		#print rect[2]
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(blur,[box],0,(0,0,255),2)
		print M['m00']
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		print cx,cy
		frame[cy-2:cy+2,cx-2:cx+2]=[0,0,255]
		cv2.waitKey(1) #==>x is to the right and y down but in the case of the matrix flip x and y directions
		cv2.imshow('frame2', frame)

		# rect = cv2.minAreaRect(hull)
		# #box = cv2.cv.boxPoints(rect)
		# box = np.int0(box)
		# cv2.drawContours(blur,[box],0,(0,0,255),2)
		# img = frame[:,:,::-1]
		# plt.imshow(img, cmap = 'gray')
		# plt.xticks([]), plt.yticks([])
		# plt.show()

	
	
	#fgmask = fgbg.apply(frame)

	#img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# ret,th1 = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV)
	# r,co=th1.shape
	# r1,c1=img.shape

	# x=0
	# y=0
	# c=0
	#th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	# for a in range(r):			#iteration over all the pixel values
	# 	for b in range(co):
	# 		val=th1[r-1][co-1]
	# 		c=c+val
	# 		x=x+ ((b+1)*val)
	# 		y=y+ ((a+1)*val)

#======================================================================================

	# x=x/c
	# y=y/c

	# X=700+(x)  		#transformation from the image space to the world space
	# Y=700+(y) 
	# print X,Y


	#gray = cv2.flip(gray,1)
	#plt.imshow(frame)
	#plt.show()
	#all the image processing comes here...it acts on a specific frame
	
	# cv2.imshow('frame',thresh)



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



print 'the last known cords were x=%d y=%d' % (cx,cy)

cap.release()
cv2.destroyAllWindows()
