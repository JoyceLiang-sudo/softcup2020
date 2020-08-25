# coding=utf-8
"""
车辆
"""
from model.util.point_util import *


def get_speed(p1, p2, p3, time):
    ppm = 288
    return calculate_average([p1, p2, p3]) / ppm * 3.6 * time


def speed_measure(tracks, time, speeds, track_kinds):
    """
    测速
    追踪编号 中点 速度
    """
    for track in tracks:
        if len(track) > track_kinds + 1:
            now_speed = get_speed(track[-1], track[-2], track[-3], time)  # 这一帧的速度
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


def show_traffic_light(image, boxes):
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
