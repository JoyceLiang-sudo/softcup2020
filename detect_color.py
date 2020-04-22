import cv2
import numpy as np
import os
image_path=''

#红跟绿有些时候容易混乱，绿灯时有时会识别为red
def detect_color(image):
    #image = cv2.imread(image_path)
    #BGR 转成 HSV
    hsv_img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    #min and max HSV values
    red_min = np.array([0,5,150])
    red_max = np.array([8,255,255])
    red_min2 = np.array([175,5,150])
    red_max2 = np.array([180,255,255])

    yello_min = np.array([26,5,150])
    yello_max = np.array([30,255,255])

    green_min = np.array([35,5,150])
    green_max = np.array([90,255,255])

    #apply red,yello,green thresh to image
    #利用cv2.inRange函数设阈值，去除背景部分
    red_thresh = cv2.inRange(hsv_img,red_min,red_max)+cv2.inRange(hsv_img,red_min2,red_max2)
    yellow_thresh = cv2.inRange(hsv_img,yello_min,yello_max)
    green_thresh = cv2.inRange(hsv_img,green_min,green_max)

    #apply blur to fix noise in thresh
    #进行中值滤波
    red_blur = cv2.medianBlur(red_thresh,5)
    yello_blur = cv2.medianBlur(yellow_thresh,5)
    green_blur = cv2.medianBlur(green_thresh,5)

    #checks which colour thresh has the most white pixels
    red = cv2.countNonZero(red_blur)
    yellow = cv2.countNonZero(yello_blur)
    green = cv2.countNonZero(green_blur)

    #the state of the light is the one with the greatest number of white pixels
    lightColor = max(red,yellow,green)
    #print(lightColor)
    
    if lightColor == red:
        return 1
    elif lightColor == yellow:
        return 2
    elif lightColor == green:
        return 3
    else :
        return 0

if __name__ == '__main__':
    detect_color(image_path)