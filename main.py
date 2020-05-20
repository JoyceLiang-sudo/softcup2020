import os
import time
import darknet
import numpy as np

from model.lane_line import draw_lane_lines
from model.plate import LPR
from model.car import get_license_plate
from model.util.point_util import *
from model.conf import conf
from model.detect_color import traffic_light
from model.zebra import Zebra, get_zebra_line, draw_zebra_line
import cv2


class Data:
    tracks = []  # 对应追踪编号的轨迹
    illegal_boxes_number = []  # 违规变道车的追踪编号
    lane_lines = []  # 车道线
    zebra_line = Zebra(0, 0, 0, 0)  # 斑马线

    init_flag = True  # 首次运行标志位

    class_names = get_names(conf.names_path)  # 标签名称
    colors = get_colors(class_names)  # 每个标签对应的颜色


class Model:
    def __init__(self):
        # 追踪器模型
        self.encoder, self.tracker = init_deep_sort()

    # darknet 模型
    netMain = darknet.load_net_custom(conf.cfg_path.encode("ascii"), conf.weight_path.encode("ascii"), 0, 1)
    metaMain = darknet.load_meta(conf.radar_data_path.encode("ascii"))
    image_width = darknet.network_width(netMain)
    image_height = darknet.network_height(netMain)
    darknet_image = darknet.make_image(image_width, image_height, 3)
    # 车牌识别模型
    plate_model = LPR(conf.plate_cascade, conf.plate_model12, conf.plate_ocr_plate_all_gru)


def YOLO():
    data = Data()
    model = Model()
    print("Starting the YOLO loop...")

    cap = cv2.VideoCapture(conf.video_path)

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        ret, frame_test = cap.read()

        if data.init_flag:
            data.zebra_line = get_zebra_line(frame_read)
            data.lane_lines = lane_line.lane_lines(frame_test, data.zebra_line)
        data.init_flag = False

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (model.image_width, model.image_height), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(model.darknet_image, frame_resized.tobytes())

        # 类别编号  置信度 (x,y,w,h)
        detections = darknet.detect_image(model.netMain, model.metaMain, model.darknet_image, thresh=conf.thresh)

        # 类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
        boxes = convert_output(detections)

        # 更新tracker
        boxes = tracker_update(boxes, frame_resized, model.encoder, model.tracker)

        # 红绿灯的颜色放在box最后面
        boxes = traffic_light(boxes, frame_rgb)

        # 把识别框映射为输入图片大小
        boxes = cast_origin(boxes, model.image_width, model.image_height, frame_rgb.shape)

        # 制作轨迹
        make_track(boxes, data.tracks)

        # 车牌识别
        boxes = get_license_plate(boxes, frame_rgb, model.plate_model)

        # 检测违规变道
        judge_illegal_change_lanes(frame_rgb, boxes, data.lane_lines, data.illegal_boxes_number)

        # 画出预测结果
        frame_rgb = draw_result(frame_rgb, boxes, data)
        draw_zebra_line(frame_rgb, data.zebra_line)
        draw_lane_lines(frame_rgb, data.lane_lines)

        # 打印预测信息
        print_info(boxes, time.time() - prev_time, data.class_names)

        # 显示图片
        out_win = "result"
        cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        cv2.imshow(out_win, frame_rgb)
        if cv2.waitKey(1) >= 0:
            cv2.waitKey(0)


if __name__ == "__main__":
    YOLO()
