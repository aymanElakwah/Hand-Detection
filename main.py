import cv2
import numpy as np
import pyautogui as mv

import HandDetector
import HandGestures

x_ratio = mv.size().width / 640.0
y_ratio = mv.size().height / 480.0
prev_center = mv.position()
prev_x = prev_center.x
prev_y = prev_center.y
cnt = 0
hand_detector = HandDetector.HandDetector()
hand_gesture = HandGestures.HandGestures()
cap = cv2.VideoCapture(0)

n = 100
threshold_front_1 = threshold_front_2 = threshold_front_3 = 0
threshold_back_1 = threshold_back_2 = threshold_back_3 = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if n > 50:
        threshold_front_1, threshold_front_2, threshold_front_3 = hand_detector.calibrate(frame)
        n -= 1
        cv2.putText(frame, text="Move your front hand", org=(50, 100), color=(255, 0, 0), fontFace=0, fontScale=1, thickness=2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    elif n > 0:
        if n == 50:
            threshold_back_1, threshold_back_2, threshold_back_3 = hand_detector.calibrate(frame, True)
        else:
            threshold_back_1, threshold_back_2, threshold_back_3 = hand_detector.calibrate(frame)
        cv2.putText(frame, text="Move your Back hand", org=(50, 100), color=(255, 0, 0), fontFace=0, fontScale=1, thickness=2)
        cv2.imshow("Frame", frame)
        n -= 1
        if cv2.waitKey(5) & 0xFF == 27:
            break
        if n == 0:
            hand_detector.set_color_threshold(threshold_front_1, threshold_front_2, threshold_front_3, threshold_back_1, threshold_back_2, threshold_back_3, 5)
    else:
        contour, center, hand_mask = hand_detector.detect_hand(frame)
        if center == (-1, -1):
            center = (int(prev_x / x_ratio), int(prev_y / y_ratio))
        cnt += 1
        if cnt % 3 == 0:
            x, y = center
            x = int(x_ratio * x)
            y = int(y * y_ratio)
            # mv.moveTo(x, y)
            prev_x = x
            prev_y = y
        img = np.zeros(frame.shape, frame.dtype)
        img[hand_mask] = frame[hand_mask]
        cv2.circle(img, center, 7, (255, 255, 255), -1)
        gest = hand_gesture.count(hand_mask, contour)
        cv2.putText(img, text=gest, org=(50, 100), color=(255, 0, 0), fontFace=0, fontScale=1, thickness=2)
        cv2.imshow("hand mask", img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
