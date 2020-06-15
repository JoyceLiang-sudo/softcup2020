# coding=utf-8
from easydict import EasyDict as edict
from PIL import ImageFont
import tensorflow as tf

# yolo 的配置文件
conf = edict()

# 视频路径，为0则打开内置摄像头
conf.video_path = "./data/video2.avi"

# 保存文件路径
conf.save_path = "./save_img/"
conf.save_path1 = "illegal_change_lane/"
conf.save_path2 = "illegal_parking/"
conf.save_path3 = "no_comity_pedestrian/"
conf.save_path4 = "retrograde_cars/"
conf.save_path5 = "running_red/"
# conf.video_path = 0

# yolo配置文件
conf.cfg_path = './data/Gaussian_yolov3_BDD.cfg'
conf.weight_path = './data/Gaussian_yolov3_BDD.weights'
conf.radar_data_path = './data/BDD.data'
conf.names_path = './data/BDD.names'

# 置信度，大于此值会被判断为真
conf.thresh = 0.3

# 字体文件
conf.fontStyle = ImageFont.truetype("font/simsun.ttc", size=20, encoding="utf-8")

# 车牌识别的模型
conf.plate_cascade = './data/plate_models/cascade.xml'
conf.plate_model12 = './data/plate_models/model12.h5'
conf.plate_ocr_plate_all_gru = './data/plate_models/ocr_plate_all_gru.h5'

# 设置tf显存占用自增长，防止显存溢出
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# deep sort 的配置文件
conf.trackerConf = edict()

conf.trackerConf.track_label = [2, 3, 4]  # 追踪器要追踪的标签号
conf.trackerConf.max_cosine_distance = 0.3
conf.trackerConf.nn_budget = None
conf.trackerConf.nms_max_overlap = 0.7
conf.trackerConf.model_filename = './data/mars-small128.pb'
