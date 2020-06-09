import os
import time
import darknet
import numpy as np
import sys

from PyQt5.QtWidgets import QApplication
from model.lane_line import draw_lane_lines, draw_stop_line
from model.plate import LPR
from model.car import get_license_plate, speed_measure, draw_speed_info
from model.util.point_util import *
from model.conf import conf
from model.detect_color import traffic_light
from model.zebra import Zebra, get_zebra_line, draw_zebra_line
from model.comity_pedestrian import judge_comity_pedestrian, Comity_Pedestrian
from model.traffic_flow import get_traffic_flow, Traffic_Flow
from model.retrograde import get_retrograde_cars
from model.running_red_lights import judge_running_car
from model.util.thread import QTThread
import cv2


class Data(object):
    tracks = []  # 对应追踪编号的轨迹
    illegal_boxes_number = []  # 违规变道车的追踪编号
    lane_lines = []  # 车道线
    stop_line = []  # 停车线
    zebra_line = Zebra(0, 0, 0, 0)  # 斑马线
    speeds = []  # 速度信息
    traffic_flow = 0
    init_flag = True  # 首次运行标志位
    retrograde_cars_number = []  # 逆行车号
    no_comity_pedestrian_cars_number = []  # 不礼让行人的车号
    true_running_car = []  # 闯红灯车辆的追踪编号
    running_car = []
    class_names = get_names(conf.names_path)  # 标签名称
    colors = get_colors(class_names)  # 每个标签对应的颜色


class Model(object):
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
    qt_thread = QTThread("qt_thread")
    qt_thread.start()
    comity_pedestrian = Comity_Pedestrian()
    traffic_flow = Traffic_Flow()
    print("Starting the YOLO loop...")

    cap = cv2.VideoCapture(conf.video_path)

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if frame_read is None:
            exit(0)
        if data.init_flag:
            data.zebra_line = get_zebra_line(frame_read)
            data.lane_lines, data.stop_line = lane_line.get_lane_lines(frame_read, data.zebra_line)
            traffic_flow.pre_time = time.time()
            data.init_flag = False

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (model.image_width, model.image_height), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(model.darknet_image, frame_resized.tobytes())

        # 类别编号  置信度 (x,y,w,h)
        detections = darknet.detect_image(model.netMain, model.metaMain, model.darknet_image, thresh=conf.thresh)

        # 类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
        boxes = convert_output(detections)

        # 更新tracker
        boxes = tracker_update(boxes, frame_resized, model.encoder, model.tracker, conf.trackerConf.track_label)

        # 红绿灯的颜色放在box最后面
        boxes = traffic_light(boxes, frame_rgb)

        # 把识别框映射为输入图片大小
        boxes = cast_origin(boxes, model.image_width, model.image_height, frame_rgb.shape)

        # 制作轨迹
        make_track(boxes, data.tracks)

        # 计算速度
        speed_measure(data.tracks, float(time.time() - prev_time), data.speeds)

        # 车牌识别
        boxes = get_license_plate(boxes, frame_rgb, model.plate_model)

        # 检测礼让行人
        data.no_comity_pedestrian_cars_number = judge_comity_pedestrian(frame_rgb, data.tracks, comity_pedestrian)

        # 检测闯红灯
        if boxes:
            data.true_running_car, data.running_car = judge_running_car(data.running_car, boxes, data.tracks,
                                                                        data.stop_line, data.lane_lines)

        # 检测违规变道
        judge_illegal_change_lanes(frame_rgb, boxes, data.lane_lines, data.illegal_boxes_number)

        # 检测车流量
        data.traffic_flow = get_traffic_flow(frame_rgb, traffic_flow, data.tracks, time.time())
        qt_thread.info("车流量为：%d" % data.traffic_flow)

        # 检测逆行车辆
        data.retrograde_cars_number = get_retrograde_cars(frame_rgb, data.lane_lines, data.tracks,
                                                          data.retrograde_cars_number)

        # 画出预测结果
        frame_rgb = draw_result(frame_rgb, boxes, data)
        draw_zebra_line(frame_rgb, data.zebra_line)
        draw_lane_lines(frame_rgb, data.lane_lines)
        draw_stop_line(frame_rgb, data.stop_line)
        # draw_speed_info(frame_rgb, data.speeds, boxes)

        # 打印预测信息
        # print_info(boxes, time.time() - prev_time, data.class_names)

        # 显示图片
        frame_rgb = cv2.resize(frame_rgb, (1640, 950), interpolation=cv2.INTER_LINEAR)
        qt_thread.set_image(frame_rgb)
        qt_thread.process_ready = True
        # print_qt_info(boxes, time.time() - prev_time, data.class_names, qt_thread)
        while not qt_thread.process_ready:
            time.sleep(0.01)
        if not qt_thread.is_alive():
            exit(0)
        # out_win = "result"
        # cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        # frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        # cv2.imshow(out_win, frame_rgb)

        key = cv2.waitKey(1)
        if key == 27:
            exit(0)
        elif key >= 0:
            cv2.waitKey(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    YOLO()
    sys.exit(app.exec_())
