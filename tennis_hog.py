import numpy as np
import cv2
import math
 
def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b
 
def rgb2hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df / mx
    v = mx
    return h, s, v
 
 
cap = cv2.VideoCapture(0)
ret, frame_large = cap.read()
scale = 0.5
frame = cv2.resize(frame_large, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
y, x, c = np.shape(frame)
 
print x, y
center = (y/2, x/2)
 
i = 0
while (True):
    # Capture frame
    ret, frame_large = cap.read()
    # cv2.imwrite("frame.jpg", frame_large)
    # frame_large = cv2.imread("frame.jpg")
 
    frame = cv2.resize(frame_large, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # cv2.resize(frame_large, (200, 100), frame)
 
    # Smooth
    frame = cv2.GaussianBlur(frame, (7, 7), 11)
    #frame = cv2.medianBlur(frame, 15)
 
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
    # define range of color in HSV
    lower_range = np.array([29, 50, 6])
    upper_range = np.array([64, 255, 255])
 
    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    # mask = 255 - mask
 
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask= mask)
 
 
    # Detect circles
    mask = cv2.GaussianBlur(mask, (5, 5), 11)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=50, param2=20, minRadius=0, maxRadius=0)
 
    if circles is not None:
    #     circles = circles[0]
        # circles = np.uint16(np.around(circles))
        # cv2.imwrite("frame.jpg", frame_large)
        for i in circles[0, :]:
            cv2.circle(res, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
 
 
    # # Show color
    # ball_color = list(frame[center[0], center[1]].astype(int))
    # print ball_color
    # cv2.circle(frame, (center[1], center[0]), 5, ball_color, 10)
 
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
 
 
    # cv2.imshow('frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
