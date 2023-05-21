import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.medianBlur(frame,5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

###
#HughCircles Detection TEST  
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,50,
                          param1=50,param2=30,minRadius=0,maxRadius=0) 
    circles = np.uint16(np.around(circles))
    #print len(circles)
    if len(circles) :
        for i in circles[0,:]:
        # draw the outer circle
            cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
            cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)

# Display the resulting frame
    cv2.imshow('Board',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
