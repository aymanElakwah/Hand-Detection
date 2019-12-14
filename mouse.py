import imutils
import cv2
import numpy as np
import threading 
import queue
import pyautogui as mv
def move_mouse(qu):
    while True:
        if(qu.empty()):
            continue
        x,y = qu.get()
        prev_x,prev_y = mv.position()
        mv.moveTo(x,y,duration=(abs(prev_x-x)+abs(prev_y-y))/2000)
def prepare_mouse():
    q = queue.Queue(100)
    t = threading.Thread(target=move_mouse,args=(q,))
    t.start()
    x = input()
    y = input()
    while x!=0 and y!=0:
        q.put([int(x),int(y)])
        x = input()
        y = input()
prepare_mouse()