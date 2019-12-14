import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        self.frame_1 = None
        self.frame_2 = None
        self.min_YCrCb = np.array([0, 133, 77], np.uint8)
        self.max_YCrCb = np.array([255, 173, 127], np.uint8)
        self.last_contour = None

    @staticmethod
    def __get_max_contour(mask, use_hull=False):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = None
        if len(contours) != 0:
            max_contour = max(contours, key=cv2.contourArea)
            if use_hull:
                max_contour = cv2.convexHull(max_contour)
        return max_contour

    @staticmethod
    def __draw_max_contour(contour, mask_shape):
        mask = np.zeros(mask_shape, np.uint8)
        if contour is None:
            return mask.astype(np.uint8)
        cv2.drawContours(mask, [contour], 0, (1, 0, 0), cv2.FILLED)
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
        max_contour = self.__get_max_contour(motion_mask, True)
        max_contour = self.check_contour(max_contour)
        motion_mask = self.__draw_max_contour(max_contour, motion_mask.shape)
        self.frame_2 = self.frame_1
        self.frame_1 = current_frame
        return motion_mask

    def get_color_mask(self, bgr_frame):
        yCrCb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCR_CB)
        skin_mask = cv2.inRange(yCrCb, self.min_YCrCb, self.max_YCrCb)
        skin_mask = cv2.dilate(skin_mask, np.ones((3, 3)), iterations=2)
        return skin_mask.astype(np.bool_)

    @staticmethod
    def get_contour_center(contour):
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return -1, -1
        x = int(moments["m10"] / moments["m00"])
        y = int(moments["m01"] / moments["m00"])
        return x, y

    def detect_hand(self, frame):
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color_mask = self.get_color_mask(frame)
        motion_mask = self.get_motion_mask(gray_frame)
        mask = motion_mask & color_mask
        contour = self.__get_max_contour(mask)
        mask = self.__draw_max_contour(contour, mask.shape)
        return self.get_contour_center(contour), mask

    def check_contour(self, contour, min_area=15000, r=1):
        if contour is None:
            if self.last_contour is not None:
                return self.last_contour
            return None
        x, y, w, h = cv2.boundingRect(contour)
        ratio = 1.0 * w / h
        if (self.last_contour is not None) and ((cv2.contourArea(contour) < min_area) or (ratio > r)):
            contour = self.last_contour
        self.last_contour = contour
        return contour
