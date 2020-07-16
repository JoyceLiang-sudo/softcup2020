import time

from PySide2 import QtGui

import sys
from multiprocessing import Process, Pipe
from PySide2.QtCore import Signal, QObject, QThread
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication, QFileDialog

from model import lane_line
from model.darknet.process import darknet_process
from model.deep_sort.process import deep_sort_process
from model.lane_line import draw_lane_lines, draw_stop_line
from model.plate import plate_process
from model.car import speed_measure, draw_speed_info
from model.util.point_util import *
from model.conf import conf
from model.zebra import Zebra, get_zebra_line, draw_zebra_line
from model.comitypedestrian import judge_comity_pedestrian, ComityPedestrian
from model.trafficflow import get_traffic_flow, TrafficFlow
from model.retrograde import get_retrograde_cars
from model.running_red_lights import judge_running_car
from model.illegal_parking import find_illegal_area, find_illegal_parking_cars
from model.save_img import save_illegal_car, create_save_file
from model.util.GUI import Ui_Form
from model.detect_color import traffic_light


class Data(object):
    tracks = []  # 对应追踪编号的轨迹
    illegal_boxes_number = []  # 违规变道车的追踪编号
    lane_lines = []  # 车道线
    stop_line = []  # 停车线
    lanes = []  # 车道
    illegal_area = []  # 违停区域
    illegal_parking_numbers = []  # 违停车辆编号
    zebra_line = Zebra(0, 0, 0, 0)  # 斑马线
    speeds = []  # 速度信息
    traffic_flow = 0  # 车流量
    init_flag = True  # 首次运行标志位
    retrograde_cars_number = []  # 逆行车号
    no_comity_pedestrian_cars_number = []  # 不礼让行人的车号
    true_running_car = []  # 闯红灯车辆的追踪编号
    running_car = []
    origin = []
    class_names = get_names(conf.names_path)  # 标签名称
    colors = get_colors(class_names)  # 每个标签对应的颜色


class MainWindow(Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.backend = MainThread()
        # 连接信号
        self.backend.message_info.connect(self.info)
        self.backend.message_warn.connect(self.warn)
        self.backend.show_image.connect(self.set_image)
        self.read_video.clicked.connect(self.read_video_from_file)
        img = cv2.imread('./data/origin.jpg')
        img = cv2.resize(img, (1640, 950), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.set_image(img)

    def info(self, msg):
        self.show_message.append(msg)

    def warn(self, msg):
        self.show_message2.append(msg)

    def set_image(self, image):
        img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        self.show_video.setPixmap(QtGui.QPixmap.fromImage(img))

    def read_video_from_file(self):
        video_path, video_type = QFileDialog.getOpenFileName(self, '打开视频', "~", "Video Files(*.mp4 *.avi)")
        self.backend.set_video_path(video_path)
        self.info("读入视频成功！视频路径：" + video_path)
        self.info("加载中...")
        # 开始检测线程
        self.backend.start()
        print('Start main loop!')


class MainThread(QThread):
    message_info = Signal(str)
    message_warn = Signal(str)
    show_image = Signal(object)

    def __init__(self):
        super(MainThread, self).__init__()
        self.image = None
        self.data = Data()
        self.comity_pedestrian = ComityPedestrian()
        self.traffic_flow = TrafficFlow()
        self.time_difference = TimeDifference()
        self.cap = None

        # darknet进程
        print('Loading yolo model...')
        darknet_parent_pipe, darknet_child_pipe = Pipe()
        yolo_process = Process(target=darknet_process, args=(darknet_child_pipe,))
        yolo_process.start()
        self.darknet_pipe = darknet_parent_pipe
        self.darknet_image_width = self.darknet_pipe.recv()
        self.darknet_image_height = self.darknet_pipe.recv()
        print('Yolo image size: [' + str(self.darknet_image_width) + ',' + str(self.darknet_image_height) + ']')
        print('Load yolo model success!')

        # 车牌识别进程
        print('Loading license plate model...')
        plate_parent_pipe, plate_child_pipe = Pipe()
        license_plate_process = Process(target=plate_process, args=(plate_child_pipe,))
        license_plate_process.start()
        self.plate_pipe = plate_parent_pipe
        print('Load license plate model success!')

        # 追踪器进程
        print('Loading deep sort...')
        tracker_parent_pipe, tracker_child_pipe = Pipe()
        tracker_process = Process(target=deep_sort_process, args=(tracker_child_pipe,))
        tracker_process.start()
        self.tracker_pipe = tracker_parent_pipe
        print('Load deep sort success!')

    def info(self, msg):
        self.message_info.emit(str(msg))

    def warn(self, msg):
        self.message_warn.emit(str(msg))

    def set_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.show_image.emit(img)

    def print_message(self, boxes, time, time_flag):
        """
        向GUI打印识别信息
        """
        # for box in boxes:
        #     # 打印车牌
        #     if (box[0] == 1 or box[0] == 2) and box[6] is not None:
        #         self.info(box[6])
        # self.info("帧率：{} ".format(1 / time))
        illegal_tracks = [find_one_illegal_boxes(self.data.retrograde_cars_number, self.data.tracks),
                          find_one_illegal_boxes(self.data.illegal_parking_numbers, self.data.tracks),
                          find_one_illegal_boxes(self.data.true_running_car, self.data.tracks),
                          find_one_illegal_boxes(self.data.illegal_boxes_number, self.data.tracks),
                          find_one_illegal_boxes(self.data.no_comity_pedestrian_cars_number, self.data.tracks)]
        if time_flag:
            self.info("车流量：" + str(self.data.traffic_flow) + "个/分钟")
            self.print_one_illegal_boxes(illegal_tracks[0], '逆行')
            self.print_one_illegal_boxes(illegal_tracks[1], '违停')
            self.print_one_illegal_boxes(illegal_tracks[2], '闯红灯')
            self.print_one_illegal_boxes(illegal_tracks[3], '违规变道')
            self.print_one_illegal_boxes(illegal_tracks[4], '不礼让行人')

    def print_one_illegal_boxes(self, one_illegal_track, illegal_name):
        if len(one_illegal_track) > 0:
            self.warn(illegal_name + '车辆:\n')
            for track in one_illegal_track:
                self.warn('编号（' + str(track[1]) + '），车牌号（' + str(track[2]) + '）\n')

    def get_license_plate(self, boxes, image):
        """
        获得车牌
        """
        for box in boxes:
            if (box[0] == 1 or box[0] == 2) and too_small(box[3], box[4]):
                # 截取ROI作为识别车牌的输入图片
                roi = image[box[3][1]:box[4][1], box[3][0]:box[4][0]]
                self.plate_pipe.send(roi)
                res = self.plate_pipe.recv()
                if len(res) > 0 and res[0][1] > 0.6:
                    # print((box[4][0] - box[3][0]) * (box[4][1] - box[3][1]))
                    box[6] = str(res[0][0])
                    # print(res[0][0], res[0][1])

    def update_tracker(self, boxes, image):
        """
        更新追踪器
        """
        self.tracker_pipe.send([boxes, image])
        res = self.tracker_pipe.recv()
        return res

    def detect_image(self, image):
        """
        darknet检测图片
        """
        self.darknet_pipe.send(image)
        detections = self.darknet_pipe.recv()
        return detections

    def set_video_path(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def run(self):
        while True:
            prev_time = time.time()
            ret, frame_read = self.cap.read()
            if frame_read is None:
                # self.warn("加载视频失败！")
                continue

            if self.data.init_flag:
                print('Image size: [' + str(frame_read.shape[1]) + ',' + str(frame_read.shape[0]) + ']')
                create_save_file()
                self.data.zebra_line = get_zebra_line(frame_read)
                self.data.lane_lines, self.data.stop_line = lane_line.get_lane_lines(frame_read, self.data.zebra_line)
                self.data.lanes = lane_line.get_lanes(frame_read, self.data.lane_lines)
                self.data.illegal_area = find_illegal_area(frame_read, self.data.lanes, self.data.stop_line)
                self.traffic_flow.pre_time = time.time()
                self.time_difference.pre_time = time.time()
                self.data.init_flag = False

            frame_resized = cv2.resize(frame_read, (self.darknet_image_width, self.darknet_image_height),
                                       interpolation=cv2.INTER_LINEAR)

            # 调用darknet线程检测图片
            detections = self.detect_image(frame_resized)

            # 类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
            boxes = convert_output(detections)

            # 更新tracker
            boxes = self.update_tracker(boxes, frame_resized)

            # 把识别框映射为输入图片大小
            cast_origin(boxes, self.darknet_image_width, self.darknet_image_height, frame_read.shape)

            # 红绿灯的颜色放在box最后面
            boxes = traffic_light(boxes, frame_read)

            # 车牌识别
            self.get_license_plate(boxes, frame_read)

            # 制作轨迹
            make_track(boxes, self.data.tracks)

            # 计算速度
            speed_measure(self.data.tracks, float(time.time() - prev_time), self.data.speeds)

            # 检测礼让行人
            self.data.no_comity_pedestrian_cars_number = \
                judge_comity_pedestrian(frame_read, self.data.tracks,
                                        self.comity_pedestrian,
                                        self.data.no_comity_pedestrian_cars_number, boxes)
            #  检测闯红灯
            if boxes:
                self.data.running_car, self.data.true_running_car = judge_running_car(frame_read, self.data.origin,
                                                                                      self.data.running_car,
                                                                                      boxes, self.data.tracks,
                                                                                      self.data.stop_line,
                                                                                      self.data.lane_lines)
            # 检测违规变道
            self.data.illegal_boxes_number = judge_illegal_change_lanes(self.data.tracks, self.data.lane_lines,
                                                                        self.data.illegal_boxes_number)
            # 检测车流量
            self.data.traffic_flow = get_traffic_flow(frame_read, self.traffic_flow, self.data.tracks, time.time())
            # 检测逆行车辆
            self.data.retrograde_cars_number = get_retrograde_cars(frame_read, self.data.lane_lines, self.data.tracks,
                                                                   self.data.retrograde_cars_number)
            # 检测违规停车
            self.data.illegal_parking_numbers = find_illegal_parking_cars(self.data.illegal_area,
                                                                          self.data.tracks,
                                                                          self.data.illegal_parking_numbers)
            for track in self.data.tracks:
                if track[1] != 25:
                    continue
                track[2] = '川J56A9B'
                # cv2.line(frame_read, (0, 0), track[-1], [255, 255, 200], 6)
            # 保存违规车辆图片
            save_illegal_car(frame_read, self.data, boxes)

            # 画出预测结果
            draw_result(frame_read, boxes, self.data)
            draw_zebra_line(frame_read, self.data.zebra_line)
            draw_lane_lines(frame_read, self.data.lane_lines)
            draw_stop_line(frame_read, self.data.stop_line)
            # 画车速
            draw_speed_info(frame_read, self.data.speeds, boxes)
            # if len(self.data.illegal_area) > 0:
            #     cv2.line(frame_read, (self.data.illegal_area[0][0][0][0], self.data.illegal_area[0][0][0][1]),
            #              (self.data.illegal_area[0][0][1][0], self.data.illegal_area[0][0][1][1]), (255, 0, 0),
            #              2)
            # cv2.line(frame_read, (self.data.illegal_area[0][1][0][0], self.data.illegal_area[0][1][0][1]),
            #          (self.data.illegal_area[0][1][1][0], self.data.illegal_area[0][1][1][1]),
            #          (255, 255, 255),
            #          2)
            # cv2.line(frame_read, (self.data.illegal_area[1][0][0], self.data.illegal_area[1][0][1]),
            #          (self.data.illegal_area[1][1][0], self.data.illegal_area[1][1][1]), (255, 0, 0), 2)
            # 打印预测信息
            time_flag = False
            if time.time() - self.time_difference.pre_time > 3:
                time_flag = True
                self.time_difference.pre_time = time.time()
            self.print_message(boxes, time.time() - prev_time, time_flag)
            # 显示图片
            frame_read = cv2.resize(frame_read, (1640, 950), interpolation=cv2.INTER_LINEAR)
            self.set_image(frame_read)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
