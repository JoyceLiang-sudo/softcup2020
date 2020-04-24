import os
import cv2
import time
import darknet
import model.plate as plate
from model.car import get_license_plate
from model.point_util import *
from model.conf import conf
from model.detect_color import traffic_light

netMain = None
metaMain = None
altNames = None


def YOLO():
    plate_model = plate.LPR(conf.plate_cascade, conf.plate_model12, conf.plate_ocr_plate_all_gru)
    class_names = get_names(conf.names_path)
    colors = get_colors(class_names)
    global metaMain, netMain, altNames
    configPath = conf.cfg_path
    weightPath = conf.weight_path
    metaPath = conf.radar_data_path
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" + os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" + os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" + os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0,
                                          1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    cap = cv2.VideoCapture(conf.video_path)
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain), 3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # 类别编号  置信度 (x,y,w,h)
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=conf.thresh)

        # 类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
        # 把识别框映射为输入图片大小
        boxes = convert_output(detections, frame_read.shape)

        # 红绿灯的颜色放在box最后面
        boxes = traffic_light(boxes, frame_rgb)

        # 车牌识别
        boxes = get_license_plate(boxes, frame_rgb, plate_model)

        # 画出预测结果
        frame_rgb = draw_result(frame_rgb, boxes, class_names, colors)

        # 打印预测信息
        # print_info(boxes, time.time() - prev_time, class_names)

        # 显示图片
        out_win = "result"
        cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        cv2.imshow(out_win, frame_rgb)
        cv2.waitKey(1)
    cap.release()


if __name__ == "__main__":
    YOLO()
