# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])
 
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
 
# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:

	# grab the current frame
	(grabbed, frame) = camera.read()
 
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break
 
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (7, 7), 11)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	#mask = cv2.erode(mask, None, iterations=2)
	#mask = cv2.dilate(mask, None, iterations=4)

	and_image=cv2.bitwise_and(frame,frame,mask = mask)


		# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None
 

	key = cv2.waitKey(20) & 0xFF

	if 1:	#key == ord("d"):
		# only proceed if at least one contour was found
		if len(cnts) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	 
			# only proceed if the radius meets a minimum size
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				#cv2.circle(frame, (int(x), int(y)), int(radius),
				#	(0, 255, 255), 2)
				#cv2.circle(frame, center, 5, (0, 0, 255), -1)
				crop_img = mask[int(y-radius):int(y+radius), int(x-radius):int(x+radius)]
				
				#crop_img = cv2.GaussianBlur(crop_img, (5, 5), 11)
				gray = crop_img #cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
				circles = cv2.HoughCircles(crop_img,cv2.HOUGH_GRADIENT,1,150,
		                  		param1=50,param2=30,minRadius=int(radius*0.6),maxRadius=int(radius*1.4)) 

				#if circles is not None:	
				#	circles = np.uint16(np.around(circles))
		
				#print len(circles)
				#if circles is not None :
				#	for i in circles[0,:]:
					# draw the outer circle
				#		cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
					# draw the center of the circle
				#		cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)

				if circles is not None:
	    				#     circles = circles[0]
					# circles = np.uint16(np.around(circles))
					# cv2.imwrite("frame.jpg", frame_large)
					for i in circles[0, :]:
		    				cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
		    				# cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

				cv2.imshow('hog',gray)			

				#cv2.imshow("cropped", crop_img)
	 			cv2.waitKey(20)
	
	

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
