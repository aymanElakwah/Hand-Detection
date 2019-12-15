import cv2
import numpy as np
import imutils
import pyautogui as mv


class MouseControl:
    def __init__(self):
        self.x_ratio = mv.size().width/640.0
        self.y_ratio = mv.size().height/480.0
        self.prev_center = mv.position()
        self.prev_x = self.prev_center.x
        self.prev_y = self.prev_center.y
        self.cnt=0
        
    def move_mouse(self,center):  
        if(center == (-1,-1)):
            center = (int(self.prev_x/self.x_ratio),int(self.prev_y/self.y_ratio))
        self.cnt+=1
        if(self.cnt%3 == 0):
            x,y = center
            x = int(self.x_ratio * x)
            y = int(y * self.y_ratio)
            mv.moveTo(x,y)
            self.prev_x = x
            self.prev_y = y
        