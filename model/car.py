# coding=utf-8
"""
车辆
"""


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


def too_small(p1, p2, size=75000):
    """
    判断矩形面积是否超过下限
    """
    # print((p2[0] - p1[0]) * (p2[1] - p1[1]))
    return (p2[0] - p1[0]) * (p2[1] - p1[1]) > size
