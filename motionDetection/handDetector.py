import numpy as np
import imutils
import cv2

a = 0.7

class handDetector:
    def __init__(self):
        self.frame_1 = None
        self.frame_2 = None
        self.m_roi = None
        self.background_model = None
        self.bg_threshold = None

    def imageDifference(self, frame):
        if self.frame_2 is None:
            self.frame_2 = frame
            return None
        elif self.frame_1 is None:
            self.frame_1 = frame
            return None
        else:
            diff_1 = frame - self.frame_1
            diff_2 = frame - self.frame_2
            self.frame_2 = self.frame_1
            self.frame_1 = frame
            result = (diff_1 > 30) & (diff_2 > 30)
            result = result.sum(axis=2).astype(np.uint8)
            result[result>0] = 255
            result = cv2.erode(result, np.ones((3, 3)), iterations=10)
            result = cv2.dilate(result, np.ones((3, 3)), iterations=30)
            result = cv2.erode(result, np.ones((3, 3)), iterations=20)
            _, contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = np.zeros(result.shape)
            if len(contours) != 0:
                max_contour = max(contours, key = cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                result[y:y+h, x:x+w] = 1
            self.m_roi = result
            return result

    def backgroundSubtraction(self, frame):
        if self.background_model is None:
            self.background_model = frame
            self.bg_threshold = np.ones(frame.shape[:2]) * 30

        diff = np.linalg.norm(self.background_model - frame, axis=2)
        result = diff >= self.bg_threshold
        mask = self.m_roi == 0
        self.background_model[mask] = a * self.background_model[mask] + (1 - a) * frame[mask] 
        self.bg_threshold[mask] = a * self.bg_threshold[mask] + 5 * (1 - a) * diff[mask]
        return result.astype(np.uint8) * 255
