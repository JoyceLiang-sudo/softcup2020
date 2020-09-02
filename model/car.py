# coding=utf-8
"""
车辆
"""
from model.plate import get_plate
from model.util.point_util import *


def get_speed(p1, p2, p3, p4, p5, time):
    ppm = 288
    return calculate_average([p1, p2, p3, p4, p5]) / ppm * 3.6 * time


def speed_measure(tracks, time, speeds, track_kinds):
    """
    测速
    追踪编号 中点 速度
    """
    for track in tracks:
        if len(track) > track_kinds + 3:
            now_speed = get_speed(track[-1], track[-2], track[-3], track[-4], track[-5], time)  # 这一帧的速度
            if now_speed * 1000 > 100:
                continue
            add_flag = False
            # 遍历速度列表
            for speed in speeds:
                if speed[0] == track[1]:
                    # 速度列表里有此物体上一帧的速度
                    speed[1] = track[-1]
                    speed[2] = now_speed * 1000
                    add_flag = True
                    break
            if not add_flag:
                # 速度列表里没有此物体的速度
                speeds.append([track[1], track[-1], now_speed * 1000])


def show_traffic_light(image, boxes):
    """
    放大交通灯显示
    """
    roi = None
    for box in boxes:
        if box[0] in [3, 4, 8, 9, 10, 11, 12]:
            p1 = (box[3][1] - 50 if (box[3][1] - 50 > 0) else box[3][1])
            p2 = box[4][1] + 50
            p3 = (box[3][0] - 50 if (box[3][0] - 50 > 0) else box[3][0])
            p4 = box[4][0] + 50
            # 截取ROI作为识别车牌的输入图片
            roi = image[p1:p2, p3:p4]
    if roi is not None:
        cv2.imshow("traffic_light", roi)
        cv2.waitKey(1)


def hypervelocity(speeds, over_speeds, boxes, max_speed=95):
    """
    判断超速
    追踪编号，速度，车牌
    """
    for speed in speeds:
        if speed[2] > max_speed:
            # 确认超速
            add_flag = True
            plate = get_plate(boxes, speed[0])
            for over_speed in over_speeds:
                if speed[0] == over_speed[0]:
                    # 已经加入超速列表
                    add_flag = False
                    over_speed[1] = speed[2]
                    over_speed[2] = plate
            if add_flag:
                over_speeds.append([speed[0], speed[2], plate])
