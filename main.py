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
    hand_mask = hand_detector.detect_hand(frame)
    img = np.zeros(frame.shape, frame.dtype)
    img[hand_mask] = frame[hand_mask]
    cv2.imshow("hand mask", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
