import handDetector
import cv2
import numpy as np

hd = handDetector.handDetector()
cap = cv2.VideoCapture(0)
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    sys.exit()

while(cap.isOpened()):
    ret, frame = cap.read()
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    diff = hd.imageDifference(frame)
    if diff is None:
        continue
    result = hd.backgroundSubtraction(frame)
    cv2.imshow("Difference", diff)
    cv2.imshow("Result", result)