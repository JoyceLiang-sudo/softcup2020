# coding=utf-8
"""
一些坐标点操作的工具
"""


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
