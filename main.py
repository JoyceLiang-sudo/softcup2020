import os
import time
import darknet
import numpy as np
import model.plate as plate
from model.car import get_license_plate
from model.util.point_util import *
from model.util import generate_detections as gdet
from model.conf import conf
from model.detect_color import traffic_light
from model.deep_sort import preprocessing, nn_matching
from model.deep_sort.detection import Detection
from model.deep_sort.tracker import Tracker

netMain = None
metaMain = None
altNames = None

xmin = 0
ymin = 0
xmax = 0
ymax = 0

def init_deep_sort():
    """
    初始化deep sort
    """
    encoder = gdet.create_box_encoder(conf.trackerConf.model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", conf.trackerConf.max_cosine_distance,
                                                       conf.trackerConf.nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker


def tracker_update(input_boxes, frame, encoder, tracker):
    """
    更新tracker
    """
    tracker_boxes = []
    for box in input_boxes:
        if box[0] == 2:
            # continue
            # 转化成 [左上x ,左上y, 宽 ,高 , 类别 ,置信度 ] 输入追踪器
            tracker_boxes.append([box[3][0], box[3][1], box[4][0] - box[3][0], box[4][1] - box[3][1], box[0], box[1]])

    if len(tracker_boxes) > 0:
        features = encoder(frame, np.array(tracker_boxes)[:, 0:4].tolist())

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(tracker_boxes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, conf.trackerConf.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()

        input_boxes = match_box(input_boxes, bbox, int(track.track_id))

    return input_boxes


def YOLO():
    global xmin, ymin, xmax, ymax
    encoder, tracker = init_deep_sort()
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

    image_width = darknet.network_width(netMain)
    image_height = darknet.network_height(netMain)

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(image_width, image_height, 3)

    while True:
        flag = True
        prev_time = time.time()
        ret, frame_read = cap.read()
        if flag :
            xmin, ymin, xmax, ymax = zebra(frame_read)
            flag = False
        draw_line(frame_read, xmin, ymin, xmax, ymax)
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        # 类别编号  置信度 (x,y,w,h)
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=conf.thresh)

        # 类别编号, 置信度, 中点坐标, 左上坐标, 右下坐标, 追踪编号(-1为未确定), 类别数据(obj)
        boxes = convert_output(detections)

        # 更新tracker
        boxes = tracker_update(boxes, frame_resized, encoder, tracker)

        # 红绿灯的颜色放在box最后面
        boxes = traffic_light(boxes, frame_rgb)

        # 把识别框映射为输入图片大小
        boxes = cast_origin(boxes, image_width, image_height, frame_rgb.shape)

        # 车牌识别
        boxes = get_license_plate(boxes, frame_rgb, plate_model)

        # 画出预测结果
        frame_rgb = draw_result(frame_rgb, boxes, class_names, colors)

        # 打印预测信息
        print_info(boxes, time.time() - prev_time, class_names)

        # 显示图片
        out_win = "result"
        cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        cv2.imshow(out_win, frame_rgb)
        if cv2.waitKey(1) >= 0:
            cv2.waitKey(0)
    cap.release()


if __name__ == "__main__":
    YOLO()
