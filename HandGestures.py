import cv2
import numpy as np
from sklearn.metrics import pairwise
import imutils


class HandGestures:
    @staticmethod
    def count(thresholded, segmented):  # frame thresholded , max contour
        thresholded = thresholded.astype(np.uint8) * 255
        if segmented is None:
            return 0
        detector = cv2.SimpleBlobDetector_create()
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 40000
        params.filterByCircularity = True
        params.minCircularity = 0.3
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresholded)
        number_holes = len(keypoints)
        for k in keypoints:
            cv2.circle(thresholded, (int(k.pt[0]), int(k.pt[1])), int(k.size / 2), (0, 0, 255), -1)

        cv2.imshow("binary", thresholded)
        chull = cv2.convexHull(segmented)
        extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
        extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
        extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])
        cX = int((extreme_left[0] + extreme_right[0]) / 2)
        cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
        distance = \
        pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
        maximum_distance = distance[distance.argmax()]
        radius = int(0.8 * maximum_distance)
        circumference = (2 * np.pi * radius)
        circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
        cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
        cnts = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)
        count = 0
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
                count += 1
        if number_holes == 0 and count == 5:
            return "Moving Mouse"
        elif number_holes == 0 and count == 2:
            return "Left Click"
        elif number_holes == 1 and (count >= 1 and count <= 3):  ##can be checked
            return "right Click"
        elif number_holes == 1 and count == 0:
            return "Double Click"
        elif number_holes == 0 and count == 1:
            return "Scroll up"
        elif number_holes == 0 and count == 0:
            return "Scroll Down"
        else:
            return "Undefined gesture"
