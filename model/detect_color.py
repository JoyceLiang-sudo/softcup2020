import cv2
import numpy as np


def detect_color(image):
    """
    红跟绿有些时候容易混乱，绿灯时有时会识别为red，基本的差不多
    """
    # image = cv2.imread(image_path)
    # BGR 转成 HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # min and max HSV values
    red_min = np.array([0, 43, 46])
    red_max = np.array([10, 255, 255])
    red_min2 = np.array([156, 43, 46])
    red_max2 = np.array([180, 255, 255])

    yello_min = np.array([26, 43, 46])
    yello_max = np.array([34, 255, 255])

    green_min = np.array([35, 43, 46])
    green_max = np.array([77, 255, 255])

    # apply red,yello,green thresh to image
    # 利用cv2.inRange函数设阈值，去除背景部分
    red_thresh = cv2.inRange(hsv_img, red_min, red_max) + cv2.inRange(hsv_img, red_min2, red_max2)
    yellow_thresh = cv2.inRange(hsv_img, yello_min, yello_max)
    green_thresh = cv2.inRange(hsv_img, green_min, green_max)

    # apply blur to fix noise in thresh
    # 进行中值滤波
    red_blur = cv2.medianBlur(red_thresh, 5)
    yello_blur = cv2.medianBlur(yellow_thresh, 5)
    green_blur = cv2.medianBlur(green_thresh, 5)

    # checks which colour thresh has the most white pixels
    red = cv2.countNonZero(red_blur)
    yellow = cv2.countNonZero(yello_blur)
    green = cv2.countNonZero(green_blur)

    # the state of the light is the one with the greatest number of white pixels
    lightColor = max(red, yellow, green)
    # print(lightColor)

    if lightColor == red:
        return 1
    elif lightColor == yellow:
        return 2
    elif lightColor == green:
        return 3
    else:
        return 0


def traffic_light(boxes, img):
    """
    识别红绿灯并存入数组
    """
    for box in boxes:
        if box[0] == 6:
            roi = img[box[3][1]:box[4][1], box[3][0]:box[4][0]]
            light_num = detect_color(roi)
            if light_num == 1:
                box[6] = 'red'
            elif light_num == 2:
                box[6] = 'yellow'
            elif light_num == 3:
                box[6] = 'green'
            else:
                box[6] = None
    return boxes
