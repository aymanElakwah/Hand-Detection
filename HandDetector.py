import cv2
import numpy as np
import imutils


class HandDetector:
    def __init__(self):
        self.frame_1 = None
        self.frame_2 = None
        self.min_YCrCb = np.array([0, 133, 77], np.uint8)
        self.max_YCrCb = np.array([255, 173, 127], np.uint8)
        self.last_contour = None
        self.haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    @staticmethod
    def __get_max_contour(mask, use_hull=False):
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        max_contour = None
        if len(contours) != 0:
            max_contour = max(contours, key=cv2.contourArea)
            if use_hull:
                max_contour = cv2.convexHull(max_contour)
        return max_contour

    @staticmethod
    def __draw_max_contour(contour, mask_shape, is_rect=False):
        mask = np.zeros(mask_shape, np.uint8)
        if contour is None:
            return mask.astype(np.bool_)
        if not is_rect:
            cv2.drawContours(mask, [contour], 0, (1, 0, 0), cv2.FILLED)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            mask[y:y + h, x:x + w] = 1
        return mask.astype(np.bool_)

    def get_motion_mask(self, current_frame, threshold=30):
        if self.frame_2 is None:
            self.frame_2 = current_frame
            return np.zeros(current_frame.shape, np.bool_)
        elif self.frame_1 is None:
            self.frame_1 = current_frame
            return np.zeros(current_frame.shape, np.bool_)
        diff1 = cv2.absdiff(current_frame, self.frame_1)
        diff2 = cv2.absdiff(current_frame, self.frame_2)
        _, binary1 = cv2.threshold(diff1, threshold, 1, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(diff2, threshold, 1, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_and(binary1, binary2)
        motion_mask = cv2.erode(motion_mask, np.ones((1, 1)), iterations=1)
        s_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        motion_mask = cv2.dilate(motion_mask, s_element, iterations=10)
        motion_mask = cv2.erode(motion_mask, s_element, iterations=9)
        max_contour = self.__get_max_contour(motion_mask, False)
        max_contour = self.check_contour(max_contour)
        motion_mask = self.__draw_max_contour(max_contour, motion_mask.shape, True)
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

    def open_holes(self, hand_mask, frame):
        frame[hand_mask == 0] = 0
        external_edge = cv2.Canny(hand_mask.astype(np.uint8) * 255, 50, 150)
        external_edge = cv2.dilate(external_edge, np.ones((5, 5)), iterations=1).astype(np.bool_)
        all_edges = cv2.Canny(frame, 50, 150).astype(np.bool_)
        internal_edge = (all_edges & external_edge.__invert__())
        internal_edge = cv2.dilate(internal_edge.astype(np.uint8), np.ones((3, 3)), iterations=0).astype(np.bool_)
        contour = self.__get_max_contour(internal_edge)
        internal_mask = self.__draw_max_contour(contour, internal_edge.shape)
        hand_mask = (hand_mask & internal_mask.__invert__())
        return hand_mask

    def detect_hand(self, frame):
        frame = cv2.GaussianBlur(frame, (11, 11), 0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.remove_face(gray_frame, frame)
        color_mask = self.get_color_mask(frame)
        motion_mask = self.get_motion_mask(gray_frame)
        mask = motion_mask & color_mask
        contour = self.__get_max_contour(mask)
        mask = self.__draw_max_contour(contour, mask.shape)
        mask = self.open_holes(mask, frame)
        return contour, self.get_contour_center(contour), mask

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

    def remove_face(self, gray, image):
        faces_rects = self.haar_cascade_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces_rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
