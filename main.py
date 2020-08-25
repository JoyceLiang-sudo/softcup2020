import time

from PySide2 import QtGui

import sys
from multiprocessing import Process, Pipe
from PySide2.QtCore import Signal, QThread
from PySide2.QtWidgets import QApplication, QFileDialog

from model import lane_line
from model.deep_sort.process import init_deep_sort, tracker_update
from model.lane_line import draw_lane_lines, draw_stop_line, make_tracks_lane, make_adjoin_matching, protect_lanes, \
    find_lane_lines_position_range, supplement_lose_lane_lines
from model.plate import plate_process
from model.car import speed_measure, draw_speed_info, show_traffic_light
from model.util.point_util import *
from model.conf import conf
from model.zebra import Zebra
from model.darknet import darknet
from model.comitypedestrian import judge_comity_pedestrian, ComityPedestrian
from model.trafficflow import get_traffic_flow, TrafficFlow
from model.retrograde import get_retrograde_cars
from model.running_red_lights import judge_running_car
from model.illegal_parking import find_illegal_area, find_illegal_parking_cars
from model.save_img import save_illegal_car, create_save_file
from model.util.GUI import Ui_Form
from model.camera import set_camera_message
from model.illegal_change_lanes import judge_illegal_change_lanes, judge_person_illegal_through_road


class Data(object):
    def __init__(self):
        self.tracks = []  # 对应追踪编号的轨迹
        self.tracks_kinds = 5  # 轨迹数组变量种类
        self.illegal_boxes_number = []  # 违规变道车的追踪编号
        self.illegal_person_number = []  # 横穿马路行人编号
        self.lane_lines = []  # 车道线
        self.lane_lines_position_range = []  # 车道线的位置范围
        self.lane_lines_spaces = []  # 车道线间距
        self.lanes_message = []  # 车道信息（从左到右）
        self.stop_line = []  # 停车线
        self.lanes = []  # 车道
        self.illegal_area = []  # 违停区域
        self.illegal_parking_numbers = []  # 违停车辆编号
        self.zebra_line = Zebra(None, None)  # 斑马线
        self.speeds = []  # 速度信息
        self.traffic_flow = 0  # 车流量
        self.init_flag = True  # 首次运行标志位
        self.retrograde_cars_number = []  # 逆行车号
        self.no_comity_pedestrian_cars_number = []  # 不礼让行人的车号
        self.running_car = [[], []]  # 闯红灯车辆的追踪编号
        self.camera_message = []  # 相机的相关信息

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
        # 设置视频
        self.backend.set_video_path(conf.video_path)
        self.backend.start()

    def info(self, msg):
        self.show_message.append(msg)

    def warn(self, msg):
        self.show_message2.append(msg)

    def set_image(self, image):
        img = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        self.show_video.setPixmap(QtGui.QPixmap.fromImage(img))

    def read_video_from_file(self):
        video_path, video_type = QFileDialog.getOpenFileName(self, '打开视频', "~", "Video Files(*.mp4 *.avi)")
        self.backend.re_init()
        self.backend.set_video_path(video_path)
        self.info("读入视频成功！视频路径：" + video_path)
        self.info("加载中...")
        # 开始检测线程
        self.backend.start()
        print('Start main loop!')


def read_template():
    big_corners = read_big_corners()
    mid_corners = read_mid_corners()
    small_corners = read_small_corners()
    straight_line = cv2.imread(conf.straight_line)
    straight_lines = [straight_line]

    corners = [big_corners, mid_corners, small_corners, straight_lines]
    # corners = straight_lines
    return corners


def read_big_corners():
    big_corner1 = cv2.imread(conf.big_corner1)
    big_corner2 = cv2.imread(conf.big_corner2)
    big_corner3 = cv2.imread(conf.big_corner3)
    big_corner4 = cv2.imread(conf.big_corner4)
    big_corner5 = cv2.imread(conf.big_corner5)
    big_corners = [big_corner1, big_corner2, big_corner3, big_corner4, big_corner5]
    return big_corners


def read_mid_corners():
    mid_corner1 = cv2.imread(conf.mid_corner1)
    mid_corner2 = cv2.imread(conf.mid_corner2)
    mid_corner3 = cv2.imread(conf.mid_corner3)
    mid_corner4 = cv2.imread(conf.mid_corner4)
    mid_corners = [mid_corner1, mid_corner2, mid_corner3, mid_corner4]
    return mid_corners


def read_small_corners():
    small_corner1 = cv2.imread(conf.small_corner1)
    small_corner2 = cv2.imread(conf.small_corner2)
    small_corner3 = cv2.imread(conf.small_corner3)
    small_corners = [small_corner1, small_corner2, small_corner3]
    return small_corners


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

        # darknet
        print('Loading yolo model...')
        self.netMain = darknet.load_net_custom(conf.cfg_path.encode("ascii"), conf.weight_path.encode("ascii"), 0, 1)
        self.metaMain = darknet.load_meta(conf.radar_data_path.encode("ascii"))
        self.darknet_image_width = darknet.network_width(self.netMain)
        self.darknet_image_height = darknet.network_height(self.netMain)
        self.darknet_image = darknet.make_image(self.darknet_image_width, self.darknet_image_height, 3)

        print('Yolo image size: [' + str(self.darknet_image_width) + ',' + str(self.darknet_image_height) + ']')
        print('Load yolo model success!')

        # 车牌识别进程
        print('Loading license plate model...')
        plate_parent_pipe, plate_child_pipe = Pipe()
        license_plate_process = Process(target=plate_process, args=(plate_child_pipe,))
        license_plate_process.start()
        self.plate_pipe = plate_parent_pipe
        print('Load license plate model success!')

        # 追踪器
        print('Loading deep sort...')
        self.encoder, self.tracker = init_deep_sort()
        print('Load deep sort success!')

    def info(self, msg):
        self.message_info.emit(str(msg))

    def warn(self, msg):
        self.message_warn.emit(str(msg))

    def set_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.show_image.emit(img)

    def re_init(self):
        self.data = Data()
        self.comity_pedestrian = ComityPedestrian()
        self.traffic_flow = TrafficFlow()
        self.time_difference = TimeDifference()
        self.encoder, self.tracker = init_deep_sort()

    def print_message(self, time_flag):
        """
        向GUI打印识别信息
        """
        illegal_tracks = [find_one_illegal_boxes(self.data.retrograde_cars_number, self.data.tracks),
                          find_one_illegal_boxes(self.data.illegal_parking_numbers, self.data.tracks),
                          find_one_illegal_boxes(self.data.running_car[1], self.data.tracks),
                          find_one_illegal_boxes(self.data.illegal_boxes_number, self.data.tracks),
                          find_one_illegal_boxes(self.data.no_comity_pedestrian_cars_number, self.data.tracks)]
        if time_flag:
            self.info("车流量：" + str(self.data.traffic_flow) + "个/分钟")
            self.print_plate(self.data.tracks)
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

    def print_plate(self, tracks):
        for track in tracks:
            if track[2] is not None:
                self.info("编号为：" + str(track[1]) + " 的车牌号为：" + track[2])

    def get_license_plate(self, boxes, image):
        """
        获得车牌
        """
        for box in boxes:
            if box[0] in [6, 13] and too_small(box[3], box[4]):
                # 截取ROI作为识别车牌的输入图片
                roi = image[box[3][1]:box[4][1], box[3][0]:box[4][0]]
                self.plate_pipe.send(roi)
                res = self.plate_pipe.recv()
                if len(res) > 0 and res[0][1] > 0.6:
                    box[6] = str(res[0][0])

    def set_video_path(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def run(self):
        while True:
            prev_time = time.time()
            ret, frame_read = self.cap.read()
            corners = read_template()
            if frame_read is None:
                self.warn("加载视频失败！")
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

            # 斑马线识别
            self.data.zebra_line.get_zebra_line(boxes, frame_read.shape)

            if self.data.init_flag:
                print('Image size: [' + str(frame_read.shape[1]) + ',' + str(frame_read.shape[0]) + ']')
                create_save_file()
                # 车道线识别
                self.data.lane_lines, self.data.stop_line = \
                    lane_line.get_lane_lines(frame_read, self.data.zebra_line.down_zebra_line, corners,
                                             self.data.init_flag)
                # 车道识别
                self.data.lanes, self.data.lane_lines = lane_line.get_lanes(frame_read,
                                                                            self.data.lane_lines, self.data.init_flag)
                # 解算车道线范围
                # self.data.lane_lines_position_range, self.data.lane_lines_spaces = find_lane_lines_position_range(
                #     self.data.lane_lines, frame_read.shape[1])
                # 获得车道信息
                self.data.lanes_message = lane_line.set_lanes_message(boxes, self.data.lanes)
                # # 检测违停区域
                # self.data.illegal_area = find_illegal_area(frame_read, self.data.lanes, self.data.stop_line)
                # 获得时间
                self.traffic_flow.pre_time = time.time()
                # 获得时间
                self.time_difference.pre_time = time.time()
                # 获得相机参数
                self.data.camera_message = set_camera_message()
                # 标志位改置
                self.data.init_flag = False

            lane_lines = lane_line.get_lane_lines(frame_read, self.data.zebra_line.down_zebra_line, None,
                                                  self.data.init_flag)
            lanes, lane_lines = lane_line.get_lanes(frame_read, lane_lines, True)
            lane_lines = make_adjoin_matching(self.data.lane_lines, lane_lines)
            # #lane_lines_position_range, lane_lines_spaces = find_lane_lines_position_range(lane_lines,
            # #                                                                              frame_read.shape[1])
            # #new_lane_lines = supplement_lose_lane_lines(self.data.lane_lines, lane_lines,
            #                                             self.data.lane_lines_position_range,
            #                                             self.data.lane_lines_spaces, lane_lines_spaces)
            lanes, pp_lane_lines = lane_line.get_lanes(frame_read, lane_lines, True)

            self.data.lane_lines, self.data.lanes = protect_lanes(self.data.lane_lines, lane_lines, self.data.lanes,
                                                                  lanes)
            # self.data.lane_lines_position_range, self.data.lane_lines_spaces = find_lane_lines_position_range(
            #     self.data.lane_lines, frame_read.shape[1])

            # 车牌识别
            self.get_license_plate(boxes, frame_read)

            # 制作轨迹
            make_track(boxes, self.data.tracks)

            # 匹配车道
            self.data.tracks = make_tracks_lane(self.data.tracks, self.data.lanes, self.data.stop_line,
                                                self.data.lanes_message)

            # 计算速度
            speed_measure(self.data.tracks, float(time.time() - prev_time), self.data.speeds, self.data.tracks_kinds)

            # 检测礼让行人
            self.data.no_comity_pedestrian_cars_number = \
                judge_comity_pedestrian(frame_read, self.data.tracks,
                                        self.comity_pedestrian,
                                        self.data.no_comity_pedestrian_cars_number, boxes, self.data.tracks_kinds)

            #  检测闯红灯
            judge_running_car(boxes, self.data.running_car, self.data.tracks, self.data.zebra_line,
                              self.data.tracks_kinds)

            # 检测违规变道
            self.data.illegal_boxes_number = judge_illegal_change_lanes(frame_read.shape[0], self.data.tracks,
                                                                        self.data.lane_lines,
                                                                        self.data.illegal_boxes_number,
                                                                        self.data.tracks_kinds)
            # 检测行人横穿马路
            self.data.illegal_person_number = judge_person_illegal_through_road(self.data.tracks,
                                                                                self.data.zebra_line.down_zebra_line,
                                                                                self.data.tracks_kinds,
                                                                                frame_read.shape[1])
            # print("illegal_boxes_number")
            # print(self.data.illegal_boxes_number)
            # 检测车流量
            self.data.traffic_flow = get_traffic_flow(frame_read, self.traffic_flow, self.data.tracks, time.time(),
                                                      self.data.tracks_kinds)

            # 检测逆行车辆
            self.data.retrograde_cars_number = get_retrograde_cars(frame_read, self.data.lane_lines, self.data.tracks,
                                                                   self.data.retrograde_cars_number,
                                                                   self.data.tracks_kinds)
            # 检测违规停车
            self.data.illegal_parking_numbers = find_illegal_parking_cars(self.data.illegal_area,
                                                                          self.data.tracks,
                                                                          self.data.illegal_parking_numbers,
                                                                          self.data.tracks_kinds)
            # 保存违规车辆图片
            save_illegal_car(frame_read, self.data, boxes)

            # corners_message = []
            # for template_img in corners:
            #     tl, br = template_demo(template_img, frame_read)
            #     corner_message = [tl, br]
            #     corners_message.append(corner_message)
            # # for corner_message in corners_message:
            # #     cv2.rectangle(frame_read, corner_message[0], corner_message[1], (0, 0, 255), 2)

            # 显示红绿灯
            # show_traffic_light(frame_read, boxes)

            # 画出预测结果
            draw_result(frame_read, boxes, self.data, self.data.tracks_kinds)
            self.data.zebra_line.draw_zebra_line(frame_read)
            draw_lane_lines(frame_read, self.data.lane_lines)
            draw_stop_line(frame_read, self.data.stop_line)
            # 画车速
            draw_speed_info(frame_read, self.data.speeds, boxes)

            # 打印预测信息
            # print_info(boxes, time.time() - prev_time)
            time_flag = False
            if time.time() - self.time_difference.pre_time > 3:
                time_flag = True
                self.time_difference.pre_time = time.time()
            self.print_message(time_flag)
            # 显示图片
            frame_read = cv2.resize(frame_read, (1640, 950), interpolation=cv2.INTER_LINEAR)
            self.set_image(frame_read)
            # out_win = "result"
            # cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
            # cv2.imshow(out_win, frame_read)
            # if cv2.waitKey(1) == 27:
            #     break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
