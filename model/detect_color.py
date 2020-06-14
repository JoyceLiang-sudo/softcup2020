import cv2
import numpy as np


def detect_color(image):
    # image = cv2.imread(image_path)
    # 准确率比上次提高了
    # RGB
    im_R = image[:, :, 0]
    im_G = image[:, :, 1]
    im_B = image[:, :, 2]

    # 平均值
    im_R_mean = np.mean(im_R)
    im_G_mean = np.mean(im_G)
    im_B_mean = np.mean(im_B)
    color = max(im_R_mean, im_G_mean, im_B_mean)

    # BGR 转成 HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # HSV色彩空间阈值
    red_min = np.array([0, 43, 46])
    red_max = np.array([8, 255, 255])
    red_min2 = np.array([156, 43, 46])
    red_max2 = np.array([180, 255, 255])

    yello_min = np.array([15, 43, 46])
    yello_max = np.array([34, 255, 255])

    green_min = np.array([35, 43, 46])
    green_max = np.array([77, 255, 255])

    # 利用cv2.inRange函数设阈值，去除背景部分
    red_thresh = cv2.inRange(hsv_img, red_min, red_max) + cv2.inRange(hsv_img, red_min2, red_max2)
    yellow_thresh = cv2.inRange(hsv_img, yello_min, yello_max)
    green_thresh = cv2.inRange(hsv_img, green_min, green_max)

    red_blur = cv2.medianBlur(red_thresh, 5)
    yellow_blur = cv2.medianBlur(yellow_thresh, 5)
    green_blur = cv2.medianBlur(green_thresh, 5)

    # 统计非零像素点数
    red = cv2.countNonZero(red_blur)
    yellow = cv2.countNonZero(yellow_blur)
    green = cv2.countNonZero(green_blur)

    # 最大值
    lightColor = max(red, yellow, green)
    # print(lightColor)

    if color == im_R_mean or lightColor == red:
        return 1
    elif color == im_G_mean or lightColor == green:
        return 2
    else:
        return 3


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
                box[6] = 'green'
            elif light_num == 3:
                box[6] = 'yellow'
            else:
                box[6] = None

    return boxes
