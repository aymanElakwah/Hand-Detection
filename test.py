import HandDetector
import cv2
import numpy as np
hand_detector = HandDetector.HandDetector()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img = hand_detector.get_color_mask(frame)
    img = img.astype(np.uint8) * 255
    cv2.imshow("binary",img)
    if cv2.waitKey(5) & 0xFF == 27:
        break