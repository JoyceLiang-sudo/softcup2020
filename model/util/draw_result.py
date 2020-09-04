# coding=utf-8
"""
在图片上画出最终结果
"""
import cv2
from model.conf import conf
from model.util.point_util import *


def print_info(boxes, time):
    """
    打印预测信息
    :param boxes: boxes
    :param time: 时间
    :return: None
    """
    print('从图片中找到 {} 个物体'.format(len(boxes)))
    count = 0
    for box in boxes:
        if box[5] != -1:
            count += 1
    print('成功追踪 {} 个物体'.format(count))
    print("所用时间：{} 秒 帧率：{} \n".format(time.__str__(), 1 / time))


def find_one_illegal_boxes(illegal_number, tracks):
    """
    找一种违规信息
    :param tracks:轨迹
    :param illegal_number:非法编号
    :return:可疑轨迹
    """
    possible_tracks = []
    for number in illegal_number:
        for track in tracks:
            if track[1] == number:
                possible_tracks.append(track)
                break
    return possible_tracks


def draw_result(image, boxes, data, track_kinds):
    """
    画出预测结果
    :param image: 图片
    :param boxes: boxes
    :param data: data类
    :param track_kinds: 轨迹结构体的种类
    :return: None
    """
    if data.tracks is None:
        return None
    for box in boxes:
        if box[0] in conf.hide_labels:
            continue
        if calculate_distance(box[2], [1250, 980]) < 20:
            continue
        box_color = data.colors[box[0]]
        box_thick = 3
        for number in data.illegal_boxes_number:
            if number == box[5]:
                box_color = [230, 100, 100]
                box_thick = 10
                break
        for number in data.drive_wrong_direction:
            if number == box[5]:
                box_color = [200, 100, 255]
                box_thick = 10
                break
        for car_person in data.no_comity_pedestrian_cars_number:
            if car_person == box[5]:
                box_color = [230, 100, 100]
                box_thick = 10
                break

        for car_light in data.running_car[1]:
            if car_light == box[5]:
                box_color = [230, 100, 100]
                box_thick = 10
                break
        for car_run in data.retrograde_cars_number:
            if car_run == box[5]:
                box_color = [230, 100, 100]
                box_thick = 10
                break
        for car_stop in data.stop_in_bus_area:
            if car_stop == box[5]:
                box_color = [230, 100, 100]
                box_thick = 10
                break
        for car_stop in data.illegal_parking_numbers:
            if car_stop == box[5]:
                box_color = [230, 100, 100]
                box_thick = 10
                break
        for car_run in data.no_comity_straight_number:
            if car_run == box[5]:
                box_color = [230, 100, 100]
                box_thick = 10
                break
        for car_run in data.illegal_person_number:
            if car_run == box[5]:
                box_color = [130, 100, 100]
                box_thick = 10
                break

        cv2.rectangle(image, box[3], box[4], box_color, box_thick)
        predicted_class = data.class_names[box[0]]
        label = '{} {:.2f}'.format(predicted_class, box[1])
        cv2.putText(image, label, (box[3][0], box[3][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, data.colors[box[0]], 2)

        # 画追踪编号
        if box[5] != -1:
            cv2.putText(image, str(box[5]), box[2], cv2.FONT_HERSHEY_SIMPLEX, 1, data.colors[box[0]], 2)
            judge_break = 0
            for track in data.tracks:
                if box[5] != track[1]:
                    continue
                i = track_kinds
                while i < len(track) - 1:
                    cv2.circle(image, track[i], 1, data.colors[box[0]], -1)
                    i = i + 1
                    judge_break = 1
                if judge_break == 1:
                    break


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
                cv2.putText(image, '{:.2f}'.format(speed[2]), (speed[1][0], speed[1][1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, [255, 255, 255], 2)
