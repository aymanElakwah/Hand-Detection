import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        self.frame_1 = None
        self.frame_2 = None
        self.min_YCrCb = np.array([0, 133, 77], np.uint8)
        self.max_YCrCb = np.array([255, 173, 127], np.uint8)

    @staticmethod
    def get_max_contour(mask, use_hull=False):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(mask.shape, np.uint8)
        if len(contours) != 0:
            max_contour = max(contours, key=cv2.contourArea)
            if use_hull:
                max_contour = cv2.convexHull(max_contour)
            cv2.drawContours(mask, [max_contour], 0, (1, 0, 0), cv2.FILLED)
        return mask.astype(np.bool_)

    def get_motion_mask(self, current_frame, threshold=30):
        if self.frame_2 is None:
            self.frame_2 = current_frame
            return np.zeros(current_frame.shape, current_frame.dtype)
        elif self.frame_1 is None:
            self.frame_1 = current_frame
            return np.zeros(current_frame.shape, current_frame.dtype)
        diff1 = cv2.absdiff(current_frame, self.frame_1)
        diff2 = cv2.absdiff(current_frame, self.frame_2)
        _, binary1 = cv2.threshold(diff1, threshold, 1, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(diff2, threshold, 1, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_and(binary1, binary2)
        motion_mask = cv2.erode(motion_mask, np.ones((1, 1)), iterations=1)
        s_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        motion_mask = cv2.dilate(motion_mask, s_element, iterations=10)
        motion_mask = cv2.erode(motion_mask, s_element, iterations=9)
        motion_mask = self.get_max_contour(motion_mask, True)
        self.frame_2 = self.frame_1
        self.frame_1 = current_frame
        return motion_mask

    def get_color_mask(self, bgr_frame):
        yCrCb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCR_CB)
        skin_mask = cv2.inRange(yCrCb, self.min_YCrCb, self.max_YCrCb)
        skin_mask = cv2.dilate(skin_mask, np.ones((3, 3)), iterations=2)
        return skin_mask.astype(np.bool_)

    def detect_hand(self, frame):
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color_mask = self.get_color_mask(frame)
        motion_mask = self.get_motion_mask(gray_frame)
        mask = motion_mask & color_mask
        mask = self.get_max_contour(mask)
        return mask
