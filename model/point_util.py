# coding=utf-8
"""
一些坐标点操作的工具
"""
from model.conf import conf
from PIL import Image, ImageDraw
import numpy
import cv2


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


def convert_output(detections, shape):
    """
    类别编号  置信度 (x,y,w,h)
    转化为
    类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
    """
    boxes = []
    for detection in detections:
        x = detection[2][0] / 512 * shape[1]
        y = detection[2][1] / 512 * shape[0]
        w = detection[2][2] / 512 * shape[1]
        h = detection[2][3] / 512 * shape[0]

        p1, p2 = convert_back(x, y, w, h)
        center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
        boxes.append([detection[0], float(format(detection[1], '.2f')), center, p1, p2, -1, None])
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
        # 打印坐标物体坐标信息
        # print(class_names[box[0]], (box[3][0], box[3][1]), (box[4][0], box[4][1]))
    print('成功追踪 {} 个物体'.format(count))
    print("所用时间：{} 秒 帧率：{} \n".format(time.__str__(), 1 / time))


def draw_result(image, boxes, class_names, colors, mode=False):
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
        for box in boxes:
            predicted_class = class_names[box[0]]
            label = '{} {:.2f}'.format(predicted_class, box[1])

            cv2.rectangle(image, box[3], box[4], colors[box[0]], 1)
            cv2.putText(image, label, (box[3][0], box[3][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box[0]], 1)

            # 画追踪编号
            if box[5] != -1:
                cv2.putText(image, str(box[5]), box[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box[0]], 1)
            # # 画车牌
            # if (box[0] == 1 or box[0] == 2) and box[6] is not None:
            #     cv2.putText(image, box[6], box[2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[box[0]], 1)
    return image
