# coding=utf-8
"""
车辆
"""
import sys
from model.util.point_util import *


def get_license_plate(boxes, image, model, thresh=0.6):
    """
    获得车牌
    类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
    """
    for box in boxes:
        if (box[0] == 1 or box[0] == 2) and too_small(box[3], box[4]):
            # 截取ROI作为识别车牌的输入图片
            roi = image[box[3][1]:box[4][1], box[3][0]:box[4][0]]
            res = model.recognize_plate(roi)
            # # print(res)
            if len(res) > 0 and res[0][1] > thresh:
                # print((box[4][0] - box[3][0]) * (box[4][1] - box[3][1]))
                box[6] = str(res[0][0])
                # print(res[0][0], res[0][1])


def too_small(p1, p2, size=60000):
    """
    判断矩形面积是否超过下限
    """
    # print((p2[0] - p1[0]) * (p2[1] - p1[1]))
    return (p2[0] - p1[0]) * (p2[1] - p1[1]) > size


def get_speed(p1, p2, time):
    return (calculate_distance(p1, p2) + sys.float_info.min) / time


def speed_measure(tracks, time, speeds):
    """
    测速
    追踪编号 中点 速度
    """
    for track in tracks:
        if len(track) > 3:
            now_speed = get_speed(track[-1], track[-2], time)  # 这一帧的速度
            add_flag = False
            # 遍历速度列表
            for speed in speeds:
                if speed[0] == track[1]:
                    # 速度列表里有此物体上一帧的速度
                    speed[1] = track[-1]
                    speed[2] = now_speed
                    add_flag = True
                    break
            if not add_flag:
                # 速度列表里没有此物体的速度
                speeds.append([track[1], track[-1], now_speed])


def draw_speed_info(image, speeds, boxes):
    """
    在图片上显示速度
    追踪编号 中点 方向 速度
    """
    for box in boxes:
        if box[5] != -1:
            for speed in speeds:
                if box[5] != speed[0]:
                    continue
                # 速度
                cv2.putText(image, '{:.2f}'.format(speed[2] * 1000), (speed[1][0], speed[1][1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, [255, 255, 255], 2)
