import time

from PySide2 import QtGui

import sys
from multiprocessing import Process, Pipe
from PySide2.QtCore import Signal, QObject, QThread
from PySide2.QtWidgets import QApplication

from model import lane_line
from model.darknet.process import darknet_process
from model.deep_sort.process import deep_sort_process
from model.lane_line import draw_lane_lines, draw_stop_line
from model.plate import plate_process
from model.car import speed_measure
from model.util.point_util import *
from model.conf import conf
from model.zebra import Zebra, get_zebra_line, draw_zebra_line
from model.comity_pedestrian import judge_comity_pedestrian, Comity_Pedestrian
from model.traffic_flow import get_traffic_flow, Traffic_Flow
from model.retrograde import get_retrograde_cars
from model.running_red_lights import judge_running_car
from model.illegal_parking import find_illegal_area, find_illegal_parking_cars
from model.save_img import save_illegal_car, create_save_file
from model.util.GUI import Ui_Form


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
        # 创建线程
        self.thread = QThread()
        self.backend.moveToThread(self.thread)
        # 开始线程
        self.thread.started.connect(self.backend.run)
        self.thread.start()

        # 等待检测线程跑完第一张图片再显示界面
        while self.backend.first_process_flag:
            time.sleep(0.1)

    def info(self, msg):
        self.show_message.append(msg)

    def warn(self, msg):
        self.show_message2.append(msg)

    def set_image(self, image):
        img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        self.show_video.setPixmap(QtGui.QPixmap.fromImage(img))


class MainThread(QObject):
    message_info = Signal(str)
    message_warn = Signal(str)
    show_image = Signal(object)

    def __init__(self):
        super(MainThread, self).__init__()
        self.image = None
        self.data = Data()
        self.comity_pedestrian = Comity_Pedestrian()
        self.traffic_flow = Traffic_Flow()
        self.cap = cv2.VideoCapture(conf.video_path)
        self.first_process_flag = True

        # darknet进程
        darknet_parent_pipe, darknet_child_pipe = Pipe()
        yolo_process = Process(target=darknet_process, args=(darknet_child_pipe,))
        yolo_process.start()
        self.darknet_pipe = darknet_parent_pipe
        self.darknet_image_width = self.darknet_pipe.recv()
        self.darknet_image_height = self.darknet_pipe.recv()

        # 车牌识别进程
        plate_parent_pipe, plate_child_pipe = Pipe()
        license_plate_process = Process(target=plate_process, args=(plate_child_pipe,))
        license_plate_process.start()
        self.plate_pipe = plate_parent_pipe

        # 追踪器进程
        tracker_parent_pipe, tracker_child_pipe = Pipe()
        tracker_process = Process(target=deep_sort_process, args=(tracker_child_pipe,))
        tracker_process.start()
        self.tracker_pipe = tracker_parent_pipe

    def info(self, msg):
        self.message_info.emit(str(msg))

    def warn(self, msg):
        self.message_warn.emit(str(msg))

    def set_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.show_image.emit(img)

    def print_message(self, boxes, time):
        """
        向GUI打印识别信息
        """
        # qt_thread.info('从图片中找到 {} 个物体'.format(len(boxes)))
        # count = 0
        for box in boxes:
            # 打印车牌
            if (box[0] == 1 or box[0] == 2) and box[6] is not None:
                self.info(box[6])
        #     # 打印坐标物体坐标信息
        #     # qt_thread.info(class_names[box[0]], (box[3][0], box[3][1]), (box[4][0], box[4][1]))
        # qt_thread.info('成功追踪 {} 个物体'.format(count))
        self.info("所用时间：{} 秒 帧率：{} \n".format(time.__str__(), 1 / time))
        illegal_boxes = [find_one_illegal_boxes(self.data.retrograde_cars_number, boxes),
                         find_one_illegal_boxes(self.data.illegal_parking_numbers, boxes),
                         find_one_illegal_boxes(self.data.true_running_car, boxes),
                         find_one_illegal_boxes(self.data.illegal_boxes_number, boxes),
                         find_one_illegal_boxes(self.data.no_comity_pedestrian_cars_number, boxes)]
        self.print_one_illegal_boxes(illegal_boxes[0], '逆行')
        self.print_one_illegal_boxes(illegal_boxes[1], '违停')
        self.print_one_illegal_boxes(illegal_boxes[2], '闯红灯')
        self.print_one_illegal_boxes(illegal_boxes[3], '违规变道')
        self.print_one_illegal_boxes(illegal_boxes[4], '不礼让行人')
        self.warn('--------------\n')

    def print_one_illegal_boxes(self, one_illegal_boxes, illegal_name):
        if len(one_illegal_boxes) <= 0:
            self.warn('无' + illegal_name + '车辆\n')
        else:
            self.warn(illegal_name + '车辆:\n')
            for box in one_illegal_boxes:
                self.warn('编号（' + str(box[5]) + '），车牌号（' + str(box[-1]) + '）\n')

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

    def run(self):
        while True:
            prev_time = time.time()
            ret, frame_read = self.cap.read()
            if frame_read is None:
                break
            if self.data.init_flag:
                create_save_file()
                self.data.zebra_line = get_zebra_line(frame_read)
                self.data.lane_lines, self.data.stop_line = lane_line.get_lane_lines(frame_read, self.data.zebra_line)
                self.data.lanes = lane_line.get_lanes(frame_read, self.data.lane_lines)
                self.data.illegal_area = find_illegal_area(frame_read, self.data.lanes, self.data.stop_line)
                self.traffic_flow.pre_time = time.time()
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
            # boxes = traffic_light(boxes, frame_rgb)

            # 制作轨迹
            make_track(boxes, self.data.tracks)

            # 车牌识别
            self.get_license_plate(boxes, frame_read)

            # 计算速度
            speed_measure(self.data.tracks, float(time.time() - prev_time), self.data.speeds)

            # 检测礼让行人
            self.data.no_comity_pedestrian_cars_number = judge_comity_pedestrian(frame_read, self.data.tracks,
                                                                                 self.comity_pedestrian,
                                                                                 self.data.no_comity_pedestrian_cars_number)
            # # 检测闯红灯
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

            # 保存违规车辆图片
            save_illegal_car(frame_read, self.data, boxes)

            # 画出预测结果
            draw_result(frame_read, boxes, self.data)
            draw_zebra_line(frame_read, self.data.zebra_line)
            draw_lane_lines(frame_read, self.data.lane_lines)
            draw_stop_line(frame_read, self.data.stop_line)
            # draw_speed_info(frame_rgb, self.data.speeds, boxes)

            # 打印预测信息
            self.print_message(boxes, time.time() - prev_time)

            # 显示图片
            frame_read = cv2.resize(frame_read, (1640, 950), interpolation=cv2.INTER_LINEAR)
            self.set_image(frame_read)
            self.first_process_flag = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
