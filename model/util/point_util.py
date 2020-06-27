# coding=utf-8
"""
一些坐标点操作的工具
"""
from model.conf import conf
from PIL import Image, ImageDraw
import numpy as np
import cv2
from model import lane_line
from model.deep_sort import preprocessing, nn_matching
from model.deep_sort.detection import Detection
from model.deep_sort.tracker import Tracker
from model.util import generate_detections as gdet


def calculate_variance(p1, p2, p3, p4):
    """
    计算两对坐标点的方差，省略开方和平均
    """
    return pow(p1[0] - p3[0], 2) + pow(p1[1] - p3[1], 2) + pow(p2[0] - p4[0], 2) + pow(p2[1] - p4[1], 2)


def calculate_average(points):
    """
    算平均距离
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
    """
    return np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def find_min(boxes, temp):
    """
    找到最佳匹配的框
    """
    while boxes[temp.index(min(temp))][5] != -1:
        # 如果最小值已经匹配到了一个框就把它剔除
        temp[temp.index(min(temp))] = float('inf')
    # 返回最佳匹配的下标
    return temp.index(min(temp))


def match_box(boxes, bbox, id):
    """
    匹配deep sort识别到的框
    boxes 类别编号 置信度 中心点 左上坐标 右下坐标 追踪编号(-1为未确定)
    """
    if len(boxes) == 0 or len(bbox) == 0:
        return []
    temp = []
    for box in boxes:
        temp.append(float(calculate_variance(box[3], box[4], (bbox[0], bbox[1]), (bbox[2], bbox[3]))))
    # 找出最小值
    if min(temp) < 1000:
        boxes[temp.index(min(temp))][5] = id
    # else:
    #     print(min(temp))
    # i = find_min(boxes, temp)
    # boxes[i][5] = id
    return boxes


def get_names(classes_path):
    """
    获得类别名称
    """
    import os
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_colors(class_names):
    """
    生成画矩形的颜色
    """
    import colorsys
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    return colors


def convert_back(x, y, w, h):
    """
    x y w h 转化成 左上坐标 右下坐标
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
    """
    boxes = []
    for detection in detections:
        p1, p2 = convert_back(detection[2][0], detection[2][1], detection[2][2], detection[2][3])
        center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        boxes.append([detection[0], float(format(detection[1], '.2f')), center, p1, p2, -1, None])
    boxes.sort()
    return boxes


def make_track(boxes, tracks):
    """
    提取中心点做轨迹
    """
    for box in boxes:
        if box[5] == -1:
            continue
        flag = 0
        for _track in tracks:
            if _track[1] == box[5]:
                _track.append(box[2])
                flag = 1
                break
        if flag == 0:
            track = [box[0], box[5], box[2]]
            tracks.append(track)


def cast_origin(boxes, origin_width, origin_height, shape):
    """
    映射为原图大小
    类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
    """
    for box in boxes:
        box[2] = (int(box[2][0] / origin_width * shape[1]), int(box[2][1] / origin_height * shape[0]))
        box[3] = (int(box[3][0] / origin_width * shape[1]), int(box[3][1] / origin_height * shape[0]))
        box[4] = (int(box[4][0] / origin_width * shape[1]), int(box[4][1] / origin_height * shape[0]))


def print_info(boxes, time, class_names):
    """
    打印预测信息
    """
    print('从图片中找到 {} 个物体'.format(len(boxes)))
    count = 0
    for box in boxes:
        if box[5] != -1:
            count += 1
        # 打印车牌
        # if (box[0] == 1 or box[0] == 2) and box[6] is not None:
        #     print(box[6])
        # 打印坐标物体坐标信息
        # print(class_names[box[0]], (box[3][0], box[3][1]), (box[4][0], box[4][1]))
    print('成功追踪 {} 个物体'.format(count))
    print("所用时间：{} 秒 帧率：{} \n".format(time.__str__(), 1 / time))


def find_one_illegal_boxes(illegal_number, boxes):
    """
    找一种违规信息
    :param illegal_number:
    :param boxes:
    :return:
    """
    possible_boxes = []
    for number in illegal_number:
        for box in boxes:
            if box[5] == number:
                possible_boxes.append(box)
                break
    return possible_boxes


def draw_result(image, boxes, data, mode=False):
    """
    画出预测结果
    """
    if mode:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        for box in boxes:
            predicted_class = data.class_names[box[0]]
            label = '{} {:.2f}'.format(predicted_class, box[1])

            draw.rectangle([tuple(box[3]), tuple(box[4])], outline=data.colors[box[0]])
            draw.text((box[3][0], box[3][1] - 5), label, data.colors[box[0]], font=conf.fontStyle)
            # 画追踪编号
            if box[5] != -1:
                draw.text(box[2], str(box[5]), data.colors[box[0]], font=conf.fontStyle)
            # 画车牌
            if (box[0] == 1 or box[0] == 2) and box[6] is not None:
                draw.text(box[2], box[6], data.colors[box[0]], font=conf.fontStyle)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    else:
        for box in boxes:
            box_color = data.colors[box[0]]
            box_color2 = data.colors[box[0]]
            box_thick = 1
            for number in data.illegal_boxes_number:
                if number == box[5]:
                    box_color = [230, 100, 100]
                    box_thick = 10
                    break
            for car_person in data.no_comity_pedestrian_cars_number:
                if car_person == box[5]:
                    box_color = [0, 0, 255]
                    box_thick = 10
                    break

            for car_light in data.true_running_car:
                if car_light == box[5]:
                    box_color = [0, 0, 255]
                    box_thick = 10
                    break
            cv2.rectangle(image, box[3], box[4], box_color, box_thick)
            # cv2.rectangle(image, box[3], box[4], box_color2, box_thick)
            predicted_class = data.class_names[box[0]]
            label = '{} {:.2f}'.format(predicted_class, box[1])
            # cv2.rectangle(image, box[3], box[4], box_color, 1)
            cv2.putText(image, label, (box[3][0], box[3][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, data.colors[box[0]], 1)

            # 画追踪编号
            if box[5] != -1:
                cv2.putText(image, str(box[5]), box[2], cv2.FONT_HERSHEY_SIMPLEX, 1, data.colors[box[0]], 2)
                judge_break = 0
                for track in data.tracks:
                    if box[5] != track[1]:
                        continue
                    i = 2
                    while i < len(track) - 1:
                        cv2.circle(image, track[i], 1, data.colors[box[0]], -1)
                        i = i + 1
                        judge_break = 1
                    if judge_break == 1:
                        break
            # # 画车牌
            # if (box[0] == 1 or box[0] == 2) and box[6] is not None:
            #     cv2.putText(image, box[6], box[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, data.colors[box[0]], 1)
            # 红绿灯
            if box[0] == 6 and box[6] is not None:
                cv2.putText(image, box[6], box[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, data.colors[box[0]], 1)


def judge_illegal_change_lanes(tracks, lane_lines, illegal_boxes_number):
    """
    判断违规变道
    """
    illegal_cars = []
    for track in tracks:
        if len(track) < 4:
            continue
        if track[0] != 2:
            continue
        for line in lane_lines:
            if judge_two_line_intersect(line[0], line[1], track[-1], track[-2]):
                illegal_cars.append(track[1])
                break
    for number1 in illegal_cars:
        flag = True
        for number2 in illegal_boxes_number:
            if number1 == number2:
                flag = False
                break
        if flag:
            illegal_boxes_number.append(number1)
    return illegal_boxes_number
    # return False


def tracker_update(input_boxes, frame, encoder, tracker, track_label):
    """
    更新tracker
    """
    tracker_boxes = []
    for box in input_boxes:
        if box[0] in track_label:
            # continue
            # 转化成 [左上x ,左上y, 宽 ,高 , 类别 ,置信度 ] 输入追踪器
            tracker_boxes.append([box[3][0], box[3][1], box[4][0] - box[3][0], box[4][1] - box[3][1], box[0], box[1]])

    if len(tracker_boxes) > 0:
        features = encoder(frame, np.array(tracker_boxes)[:, 0:4].tolist())

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(tracker_boxes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, conf.trackerConf.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            input_boxes = match_box(input_boxes, bbox, int(track.track_id))


def init_deep_sort():
    """
    初始化deep sort
    """
    encoder = gdet.create_box_encoder(conf.trackerConf.model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", conf.trackerConf.max_cosine_distance,
                                                       conf.trackerConf.nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker


# 计算斜率
def get_slope(point1, point2):
    point_1 = point1
    point_2 = point2
    if point_1[0] == point_2[0]:
        point_1 = [point_1[0] + 1, point_1[1]]
    return np.fabs((point_2[1] - point_1[1]) / (point_2[0] - point_1[0]))


# 判断两条线段相交
def judge_two_line_intersect(p1, p2, p3, p4):
    flag1 = (p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])
    flag2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    flag3 = (p4[0] - p3[0]) * (p2[1] - p3[1]) - (p4[1] - p3[1]) * (p2[0] - p3[0])
    flag4 = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])
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
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
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
