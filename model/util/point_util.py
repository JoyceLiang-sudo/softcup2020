# coding=utf-8
"""
一些坐标点操作的工具
"""
from model.conf import conf
from PIL import Image, ImageDraw
import numpy
import cv2
from model import lane_line


def calculate_variance(p1, p2, p3, p4):
    """
    计算两对坐标点的方差，省略开方和平均
    """
    return pow(p1[0] - p3[0], 2) + pow(p1[1] - p3[1], 2) + pow(p2[0] - p4[0], 2) + pow(p2[1] - p4[1], 2)


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
    i = find_min(boxes, temp)
    boxes[i][5] = id
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
            if _track[0] == box[5]:
                _track.append(box[2])
                flag = 1
                break
        if flag == 0:
            track = []
            track.append(box[5])
            track.append(box[2])
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
    return boxes


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


def draw_result(image, boxes, class_names, colors, tracks, illegal_boxes_number, mode=False):
    """
    画出预测结果
    """
    if mode:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)

        for box in boxes:
            predicted_class = class_names[box[0]]
            label = '{} {:.2f}'.format(predicted_class, box[1])

            draw.rectangle([tuple(box[3]), tuple(box[4])], outline=colors[box[0]])
            draw.text((box[3][0], box[3][1] - 5), label, colors[box[0]], font=conf.fontStyle)
            # 画追踪编号
            if box[5] != -1:
                draw.text(box[2], str(box[5]), colors[box[0]], font=conf.fontStyle)
            # 画车牌
            if (box[0] == 1 or box[0] == 2) and box[6] is not None:
                draw.text(box[2], box[6], colors[box[0]], font=conf.fontStyle)
        image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    else:
        illegal_flag = False
        for box in boxes:
            box_color = colors[box[0]]
            box_thick = 1
            for number in illegal_boxes_number:
                if number == box[5]:
                    illegal_flag = True
                    box_color = [230, 100, 100]
                    box_thick = 10
                    break
            cv2.rectangle(image, box[3], box[4], box_color, box_thick)
            predicted_class = class_names[box[0]]
            label = '{} {:.2f}'.format(predicted_class, box[1])
            # cv2.rectangle(image, box[3], box[4], box_color, 1)
            cv2.putText(image, label, (box[3][0], box[3][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box[0]], 1)

            # 画追踪编号
            if box[5] != -1:
                cv2.putText(image, str(box[5]), box[2], cv2.FONT_HERSHEY_SIMPLEX, 1, colors[box[0]], 2)
                judgeBreak = 0
                for track in tracks:
                    if box[5] != track[0]:
                        continue
                    i = 1
                    while i < len(track) - 1:
                        cv2.circle(image, track[i], 1, colors[box[0]], -1)
                        i = i + 1
                        judgeBreak = 1
                    if judgeBreak == 1:
                        break
            # # 画车牌
            # if (box[0] == 1 or box[0] == 2) and box[6] is not None:
            #     cv2.putText(image, box[6], box[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box[0]], 1)
            # 红绿灯
            if box[0] == 6 and box[6] is not None:
                cv2.putText(image, box[6], box[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box[0]], 1)
    return image


def judge_illegal_change_lanes(img, boxes, lane_lines, illegal_boxes_number):
    for box in boxes:
        for line in lane_lines:
            mid_point = [int((box[3][0] + box[4][0]) / 2), box[4][1]]
            k1 = (line[0][1] - line[1][1]) / (line[0][0] - line[1][0])
            if mid_point[0] == line[1][0]:
                mid_point[0] = mid_point[0] + 1
            k2 = (mid_point[1] - line[1][1]) / (mid_point[0] - line[1][0])
            if abs(k1 - k2) < 0.05 and box[4][1] > line[0][1]:
                if box[5] != -1:
                    illegal_boxes_number.append(box[5])
                # cv2.line(img, (mid_point[0], mid_point[1]), (line[1][0], line[1][1]), [255, 0, 255], 4)
                # return True
    # return False


