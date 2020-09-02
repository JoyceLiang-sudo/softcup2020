# coding=utf-8
"""
一些坐标点操作的工具
"""
import numpy as np
import cv2
from model.conf import *


class TimeDifference:
    pre_time = 0


def convert_back(x, y, w, h):
    """
    x y w h 转化成 左上坐标 右下坐标
    :param x: 左上横坐标
    :param y: 左上纵坐标
    :param w: 宽度
    :param h: 高度
    :return: 左上坐标 右下坐标
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))

    xmin = (xmin if (xmin > 0) else 0)
    xmax = (xmax if (xmax > 0) else 0)
    ymin = (ymin if (ymin > 0) else 0)
    ymax = (ymax if (ymax > 0) else 0)
    return (xmin, ymin), (xmax, ymax)


def convert_output(detections):
    """
    类别编号  置信度 (x,y,w,h)
    转化为
    类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
    :param detections: 类别编号  置信度 (x,y,w,h)
    :return: None
    """
    boxes = []
    for detection in detections:
        p1, p2 = convert_back(detection[2][0], detection[2][1], detection[2][2], detection[2][3])
        center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        boxes.append([detection[0], float(format(detection[1], '.2f')), center, p1, p2, -1, None])
    boxes.sort()
    return boxes


def cast_origin(boxes, origin_width, origin_height, shape):
    """
    映射为原图大小
    类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
    :param boxes: boxes
    :param origin_width: 原始宽
    :param origin_height: 原始长
    :param shape: 现在长宽
    :return: None
    """
    for box in boxes:
        box[2] = (int(box[2][0] / origin_width * shape[1]), int(box[2][1] / origin_height * shape[0]))
        box[3] = (int(box[3][0] / origin_width * shape[1]), int(box[3][1] / origin_height * shape[0]))
        box[4] = (int(box[4][0] / origin_width * shape[1]), int(box[4][1] / origin_height * shape[0]))


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


# 计算斜率
def get_slope(point1, point2):
    """
    计算斜率
    :param point1:点1
    :param point2:点2
    :return:斜率
    """
    point_1 = point1
    point_2 = point2
    if point_1[0] == point_2[0]:
        point_1 = [point_1[0] + 1, point_1[1]]
    return np.fabs((point_2[1] - point_1[1]) / (point_2[0] - point_1[0]))


# 判断两条线段相交
def judge_two_line_intersect(p1, p2, p3, p4):
    flag1 = ((p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])) * 0.001
    flag2 = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])) * 0.001
    flag3 = ((p4[0] - p3[0]) * (p2[1] - p3[1]) - (p4[1] - p3[1]) * (p2[0] - p3[0])) * 0.001
    flag4 = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) * 0.001
    if flag1 * flag2 < 0 and flag3 * flag4 < 0:
        return True
    return False


def get_intersection_point(line1, line2):
    """
    解两条直线交点
    """
    if line1[0][0] == line1[1][0]:
        line1[0][0] = line1[0][0] + 1
    if line2[0][0] == line2[1][0]:
        line2[0][0] = line2[0][0] + 1
    point = []
    k1 = (line1[0][1] - line1[1][1]) / (line1[0][0] - line1[1][0])
    b1 = line1[0][1] - k1 * line1[0][0]
    k2 = (line2[0][1] - line2[1][1]) / (line2[0][0] - line2[1][0])
    b2 = line2[0][1] - k2 * line2[0][0]
    if k1 == k2:
        x = (b2 - b1) / (k1 - k2 + 1)
    else:
        x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    # print("k1")
    # print(k1)
    # print("b1")
    # print(b1)
    # print("k2")
    # print(k2)
    # print("b2")
    # print(b2)
    # print("x")
    # print(x)
    # print("y")
    # print(y)
    point.append(int(x))
    point.append(int(y))
    return point


def judge_point_line_position(point, line):
    """
    判断点在线的相对位置，-1为左，0为线上，1为右
    (line[0]为上点，line[1]为下点)
    """
    flag = (point[0] - line[1][0]) * (line[0][1] - line[1][1]) - (line[0][0] - line[1][0]) * (point[1] - line[1][1])
    if flag == 0:
        return 0
    if flag < 0:
        return -1
    return 1


def judge_point_in_lines(point, line1, line2):
    """
    判断点是否在两个线之间
    :param point: 判断点
    :param line1: 线1
    :param line2: 线2
    :return: True-点在两条线之间，False-点不在两条线之间
    """
    if judge_point_line_position(point, line1) == -1 and judge_point_line_position(point, line2) == 1:
        return True
    return False


def find_real_numbers(pre_numbers, now_numbers):
    """
    和类成员合并，取消相同项
    """
    for number1 in now_numbers:
        flag = True
        for number2 in pre_numbers:
            if number1 == number2:
                flag = False
                break
        if flag:
            pre_numbers.append(number1)
    return pre_numbers


def judge_stop(track):
    """
    判断是否静止
    :param track: 轨迹
    :return: True-静止，False-运动
    """
    if len(track) >= 7:
        if calculate_average_deviation([track[-1], track[-2], track[-3], track[-4], track[-5]]) > 5:
            return False
        return True


def calculate_two_point_distance(x1, y1, x2, y2):
    """
    计算两点间距离
    :param x1: 点1横坐标
    :param y1: 点1纵坐标
    :param x2: 点2横坐标
    :param y2: 点2纵坐标
    :return: 距离
    """
    return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def calculate_extremum_side(rect, side_kind):
    """
    计算矩形极值边
    :param rect:输入矩形
    :param side_kind:需要得到边的种类，True-长边，False-短边
    :return: 极值矩形边长度
    """
    rect_points = cv2.boxPoints(rect)
    rect_points = np.int0(rect_points)
    side1 = calculate_distance(rect_points[0], rect_points[1])
    side2 = calculate_distance(rect_points[1], rect_points[2])
    lone_side = side1 if side1 > side2 else side2
    short_side = side1 if side1 < side2 else side2
    if side_kind:
        return lone_side
    return short_side


def calculate_extremum_lines(lines):
    max_length = 0
    for line in lines:
        length = calculate_distance(line[0], line[1])
        if length > max_length:
            max_length = length
    return max_length


def template_demo(template_img, src_img):
    """
    模板匹配
    :param template_img: 截取图像
    :param src_img: 原图像
    :return: tl-左上点，br-右下点
    """
    method = cv2.TM_CCORR_NORMED
    th, tw = template_img.shape[:2]
    result = cv2.matchTemplate(src_img, template_img, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tl = max_loc
    br = (tl[0] + tw, tl[1] + th)
    return tl, br


def calculate_variance(p1, p2, p3, p4):
    """
    计算两对坐标点的方差，省略开方和平均
    :param p1: 点1
    :param p2: 点2
    :param p3: 点3
    :param p4: 点4
    :return: 方差
    """
    return pow(p1[0] - p3[0], 2) + pow(p1[1] - p3[1], 2) + pow(p2[0] - p4[0], 2) + pow(p2[1] - p4[1], 2)


def calculate_average(points):
    """
    计算平均距离
    :param points: 点集
    :return: 平均距离
    """
    if len(points) <= 1:
        return 0
    pre_point = points[0]
    average = 0
    flag = False
    for point in points:
        if not flag:
            flag = True
            continue
        now_point = point
        average = average + calculate_distance(pre_point, now_point)
        pre_point = point
    average = average / (len(points) - 1)
    return average


def calculate_average_deviation(points):
    """
    计算平均差
    :param points: 点集
    :return: 平均差
    """
    if len(points) <= 1:
        return 0
    average = calculate_average(points)
    average_deviation = 0
    pre_point = points[0]
    flag = False
    for point in points:
        if not flag:
            flag = True
            continue
        now_point = point
        average_deviation = average_deviation + np.fabs(calculate_distance(pre_point, now_point) - average)
        pre_point = point
    average_deviation = average_deviation / (len(points) - 1)
    return average_deviation


def calculate_distance(p1, p2):
    """
    计算两个点的距离
    :param p1: 点1
    :param p2: 点2
    :return: 距离
    """
    return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def too_small(p1, p2, size=60000):
    """
    判断矩形面积是否超过下限
    :param p1: 左上点
    :param p2: 右下点
    :param size: 阈值下限
    :return: True-超过，False-没超过
    """
    return (p2[0] - p1[0]) * (p2[1] - p1[1]) > size
