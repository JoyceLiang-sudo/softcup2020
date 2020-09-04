# coding=utf-8
import threading
import time

import cv2

from model.darknet import darknet
from model.conf import conf
from model.deep_sort.process import init_deep_sort, tracker_update
from model.util.point_util import convert_output, cast_origin


def get_names(classes_path):
    """
    获得类别名称
    :param classes_path: 类别路径
    :return: 类别名称
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
    :param class_names: 类名称
    :return: 颜色
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


class DarknetThread(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self)
        self.name = name
        self.is_loop = True
        self.video_flag = False
        self.cap = None
        self.queue = queue
        print('Loading yolo corner...')
        self.netMain = darknet.load_net_custom(conf.cfg_path.encode("ascii"), conf.weight_path.encode("ascii"), 0, 1)
        self.metaMain = darknet.load_meta(conf.radar_data_path.encode("ascii"))
        self.darknet_image_width = darknet.network_width(self.netMain)
        self.darknet_image_height = darknet.network_height(self.netMain)
        self.darknet_image = darknet.make_image(self.darknet_image_width, self.darknet_image_height, 3)
        print('Yolo image size: [' + str(self.darknet_image_width) + ',' + str(self.darknet_image_height) + ']')
        print('Load yolo corner success!')

        # 追踪器
        print('Loading deep sort...')
        self.encoder, self.tracker = init_deep_sort()
        print('Load deep sort success!')
        self.queue.put([None, None, False])

    def run(self):
        while self.is_loop:
            if not self.video_flag:
                continue

            ret, frame_read = self.cap.read()

            if frame_read is None:
                self.video_flag = False
                self.queue.put([None, None, False])
                continue

            frame_resized = cv2.resize(frame_read, (self.darknet_image_width, self.darknet_image_height),
                                       interpolation=cv2.INTER_LINEAR)
            # 检测图片
            darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

            # 类别编号  置信度 (x,y,w,h)
            detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=conf.thresh)

            # 类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
            boxes = convert_output(detections)

            # 更新tracker
            boxes = tracker_update(boxes, frame_resized, self.encoder, self.tracker, conf.trackerConf.track_label)

            # 把识别框映射为输入图片大小
            cast_origin(boxes, self.darknet_image_width, self.darknet_image_height, frame_read.shape)

            self.queue.put([frame_read, boxes, True])

    def re_init(self, path):
        self.video_flag = False
        time.sleep(0.2)
        self.cap = cv2.VideoCapture(path)
        self.encoder, self.tracker = init_deep_sort()
        self.queue.queue.clear()
        self.video_flag = True

    def stop(self):
        """
        结束线程
        """
        self.is_loop = False
