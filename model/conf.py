# coding=utf-8
from easydict import EasyDict as edict
import tensorflow as tf

# yolo 的配置文件
conf = edict()

# 视频路径，为0则打开内置摄像头
# conf.video_path = "./data/大路口.mp4"

# 保存文件路径
conf.save_path = "./save_img/"
conf.save_path1 = "illegal_change_lane/"
conf.save_path2 = "illegal_parking/"
conf.save_path3 = "no_comity_pedestrian/"
conf.save_path4 = "retrograde_cars/"
conf.save_path5 = "running_red/"

# yolo配置文件
conf.cfg_path = './data/SoftCup.cfg'
conf.weight_path = './data/SoftCup_final.weights'
conf.radar_data_path = './data/SoftCup.data'
conf.names_path = './data/SoftCup.names'
conf.hide_labels = [14]  # 不在最终结果上画出来的类别编号

# 置信度，大于此值会被判断为真
conf.thresh = 0.3

# 车牌识别的模型
conf.plate_cascade = './data/plate_models/cascade.xml'
conf.plate_model12 = './data/plate_models/model12.h5'
conf.plate_ocr_plate_all_gru = './data/plate_models/ocr_plate_all_gru.h5'

# 保存模板路径
conf.big_corner1 = "./data/corner/big-corner1.PNG"
conf.big_corner2 = "./data/corner/big-corner2.PNG"
conf.big_corner3 = "./data/corner/big-corner3.PNG"
conf.big_corner4 = "./data/corner/big-corner4.PNG"
conf.big_corner5 = "./data/corner/big-corner5.PNG"

conf.mid_corner1 = "./data/corner/mid-corner1.PNG"
conf.mid_corner2 = "./data/corner/mid-corner2.PNG"
conf.mid_corner3 = "./data/corner/mid-corner3.PNG"
conf.mid_corner4 = "./data/corner/mid-corner4.PNG"

conf.small_corner1 = "./data/corner/small-corner1.PNG"
conf.small_corner2 = "./data/corner/small-corner2.PNG"
conf.small_corner3 = "./data/corner/small-corner3.PNG"

conf.straight_line = "./data/corner/straight-line.PNG"

# 设置tf显存占用自增长，防止显存溢出
physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

# deep sort 的配置文件
conf.trackerConf = edict()

conf.trackerConf.track_label = [2, 6, 7, 13]  # 追踪器要追踪的标签号
conf.trackerConf.max_cosine_distance = 0.3
conf.trackerConf.nn_budget = None
conf.trackerConf.nms_max_overlap = 0.7
conf.trackerConf.model_filename = './data/mars-small128.pb'
