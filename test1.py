import cv2
import numpy as np
import matplotlib.pyplot as plt

cap=cv2.VideoCapture('vid.mp4')

#if not cap.isOpened(): cap.open()

cap.get(5)

while(True):
	ret,frame=cap.read()
	#gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#plt.subplot(121);plt.imshow(frame)
	#plt.show()
	#cv2.imshow('frame',gray)


	if cv2.waitKey(10) & 0xFF == ord('q'):
		break




cap.release()
cv2.destroyAllWindows()
