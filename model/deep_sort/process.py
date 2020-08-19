# coding=utf-8
import numpy as np

from model.deep_sort.detection import Detection
from model.conf import conf
from model.deep_sort import preprocessing, nn_matching, generate_detections as gdet
from model.deep_sort.tracker import Tracker
from model.util.point_util import calculate_variance


def init_deep_sort():
    """
    初始化deep sort
    """
    encoder = gdet.create_box_encoder(conf.trackerConf.model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", conf.trackerConf.max_cosine_distance,
                                                       conf.trackerConf.nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker


def tracker_update(input_boxes, frame, encoder, tracker, track_label):
    """
    更新tracker
    """
    tracker_boxes = []
    for box in input_boxes:
        if box[0] in track_label:
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

        result_box = input_boxes
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            result_box = match_box(result_box, bbox, int(track.track_id))
        return result_box
    else:
        return []


def match_box(boxes, bbox, id):
    """
    匹配deep sort识别到的框
    boxes 类别编号 置信度 中心点 左上坐标 右下坐标 追踪编号(-1为未确定)
    """
    if len(boxes) == 0 or len(bbox) == 0:
        return []
    temp = []
    for box in boxes:
        temp.append(float(calculate_variance(box[3], box[4], (bbox[0], bbox[1]), (bbox[2], bbox[3]))))
    # 找出最小值
    if min(temp) < 1000:
        boxes[temp.index(min(temp))][5] = id
    # else:
    #     print(min(temp))
    # i = find_min(boxes, temp)
    # boxes[i][5] = id
    return boxes


def deep_sort_process(pipe):
    """
    车牌识别进程
    """
    encoder, tracker = init_deep_sort()
    while True:
        message = pipe.recv()
        if message[2]:
            encoder, tracker = init_deep_sort()
        res = tracker_update(message[0], message[1], encoder, tracker, conf.trackerConf.track_label)
        pipe.send(res)
