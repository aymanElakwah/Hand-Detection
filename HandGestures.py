import math
import cv2


class HandGestures:
    def Recognize(self, frame, cnt):
        if cnt is None:
            return "none"
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        hull = cv2.convexHull(cnt)
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
        if (areacnt == 0):
            return "none"
        arearatio = ((areahull - areacnt) / areacnt) * 100
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        if (defects is None):
            return "none"
        l = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
            d = (2 * ar) / a
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(frame, far, 3, [255, 0, 0], -1)
            cv2.line(frame, start, end, [0, 255, 0], 2)
        l += 1
        if l == 1:
            if areacnt < 2000:
                return "none"
            else:
                if arearatio < 12:
                    return "Scroll Down"
                elif arearatio < 17.5:
                    return "Scroll Up"
                else:
                    return "none"
        elif l == 2:
            return "Left Click"
        elif l == 3:
            return "Right Click"
        elif l == 4:
            return "Double Click"
        elif l == 5:
            return "Moving Mouse"
        else:
            return "none"
