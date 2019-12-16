import cv2
import numpy as np
import HandDetector
import HandGestures
import mouse

hand_detector = HandDetector.HandDetector()
hand_gesture = HandGestures.HandGestures()
mouse_moving = mouse.MouseControl()
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("/hme/ayman/Desktop/video.mp4")
n = 0
half_n = int(n / 2)
threshold_front_1 = threshold_front_2 = threshold_front_3 = 0
threshold_back_1 = threshold_back_2 = threshold_back_3 = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    if n > half_n:
        threshold_front_1, threshold_front_2, threshold_front_3 = hand_detector.calibrate(frame)
        n -= 1
        cv2.putText(frame, text="Move your front hand", org=(50, 100), color=(255, 0, 0), fontFace=0, fontScale=1,
                    thickness=2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    elif n > 0:
        if n == half_n:
            threshold_back_1, threshold_back_2, threshold_back_3 = hand_detector.calibrate(frame, True)
        else:
            threshold_back_1, threshold_back_2, threshold_back_3 = hand_detector.calibrate(frame)
        cv2.putText(frame, text="Move your Back hand", org=(50, 100), color=(255, 0, 0), fontFace=0, fontScale=1,
                    thickness=2)
        cv2.imshow("Frame", frame)
        n -= 1
        if cv2.waitKey(5) & 0xFF == 27:
            break
        if n == 0:
            hand_detector.set_color_threshold(threshold_front_1, threshold_front_2, threshold_front_3, threshold_back_1,
                                              threshold_back_2, threshold_back_3, 50)
    else:
        contour, center, hand_mask = hand_detector.detect_hand(frame)
        mouse_moving.move_mouse(center)
        img = np.zeros(frame.shape, frame.dtype)
        img[hand_mask] = frame[hand_mask]
        cv2.circle(img, center, 7, (255, 255, 255), -1)
        gest = hand_gesture.Recognize(img, contour)
        # mouse_moving.mouse_action(gest)
        cv2.putText(img, text=str(gest), org=(50, 100), color=(255, 0, 0), fontFace=0, fontScale=1, thickness=2)
        cv2.imshow("hand mask", img)
        cv2.imshow("mask", hand_mask.astype(np.uint8) * 255)
        if cv2.waitKey(5) & 0xFF == 27:
            break
