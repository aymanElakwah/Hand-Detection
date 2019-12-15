import HandDetector
import imutils
import HandGestures
import cv2
import numpy as np
import threading 
import pyautogui as mv
import mouse
hand_detector = HandDetector.HandDetector()
hand_gesture = HandGestures.HandGestures()
mouse_moving = mouse.MouseControl()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    contour, center, hand_mask = hand_detector.detect_hand(frame)
    mouse_moving.move_mouse(center)
    img = np.zeros(frame.shape, frame.dtype)
    img[hand_mask] = frame[hand_mask]
    cv2.circle(img, center, 7, (255, 255, 255), -1)
    gest = hand_gesture.count(hand_mask, contour)
    cv2.putText(img, text=str(gest), org=(50, 100), color=(255, 0, 0), fontFace=0, fontScale=1, thickness=2)
    cv2.imshow("hand mask", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
