import cv2
import numpy as np
from sklearn.metrics import pairwise
import imutils
import math

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

        # cv2.imshow("binary", thresholded)
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
        elif number_holes == 1 and (count >= 1 or count <= 3):  ##can be checked
            return "right Click"
        elif number_holes == 1 and count == 0:
            return "Double Click"
        elif number_holes == 0 and count == 1:
            return "Scroll up"
        elif number_holes == 0 and count == 0:
            return "Scroll Down"
        else:
            return "Undefined gesture"
    
    def testing(self,frame,cnt):
        print(cnt)
        if(cnt is None):
            return
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
        hull = cv2.convexHull(cnt)
    
        #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
        
        #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100

        #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        # l = no. of defects
        l=0
        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(frame, far, 3, [255,0,0], -1)
            #draw lines around hand
            cv2.line(frame,start, end, [0,255,0], 2)
        l+=1
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<12:
                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                elif arearatio<17.5:
                    cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)    
                else:
                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)     
        elif l==2:
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
            if arearatio<27:
                cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif l==4:
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==5:
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            