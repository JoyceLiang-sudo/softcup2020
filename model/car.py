# coding=utf-8
"""
车辆
"""
import sys
import cv2


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
            # print(res)
            if len(res) > 0 and res[0][1] > thresh:
                # print((box[4][0] - box[3][0]) * (box[4][1] - box[3][1]))
                box[6] = str(res[0][0])
                # print(res[0][0], res[0][1])

    return boxes


def too_small(p1, p2, size=60000):
    """
    判断矩形面积是否超过下限
    """
    # print((p2[0] - p1[0]) * (p2[1] - p1[1]))
    return (p2[0] - p1[0]) * (p2[1] - p1[1]) > size


def get_direct(p1, p2):
    """
    判断运动方向
    0-上 1-下 2-左 3-右
    """
    up_down = p1[1] - p2[1]
    left_right = p1[0] - p2[0]
    if abs(up_down) > abs(left_right):
        return 1 if up_down > 0 else 0
    else:
        return 3 if left_right > 0 else 2


def get_speed(p1, p2, direct, time):
    if direct < 2:
        speed = (abs(p1[1] - p2[1]) + sys.float_info.min) / time
    else:
        speed = (abs(p1[0] - p2[0]) + sys.float_info.min) / time
    return speed


def speed_measure(tracks, time, speeds):
    """
    测速
    追踪编号 中点 方向 速度
    """
    for track in tracks:
        if len(track) > 2:
            now_direct = get_direct(track[-1], track[1])  # 这一帧的方向
            now_speed = get_speed(track[-1], track[-2], now_direct, time)  # 这一帧的速度
            add_flag = False
            # 遍历速度列表
            for speed in speeds:
                if speed[0] == track[0]:
                    # 速度列表里有此物体上一帧的速度
                    last_direct = speed[2]
                    if last_direct == now_direct:
                        # 方向和上一帧的方向相同才会更新速度
                        speed[1] = track[-1]
                        speed[3] = now_speed
                    else:
                        # 方向不同，重新计算速度，清空列表中此元素的速度信息
                        speeds.remove(speed)
                    add_flag = True
                    break
            if not add_flag:
                # 速度列表里没有此物体的速度和方向
                speeds.append([track[0], track[-1], now_direct, now_speed])


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
                if speed[2] == 0:
                    direct = 'up'
                elif speed[2] == 1:
                    direct = 'down'
                elif speed[2] == 2:
                    direct = 'left'
                else:
                    direct = 'right'
                # 方向
                cv2.putText(image, direct, (speed[1][0], speed[1][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            [255, 255, 255], 2)
                # 速度
                cv2.putText(image, '{:.2f}'.format(speed[3] * 1000), (speed[1][0], speed[1][1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, [255, 255, 255], 2)
