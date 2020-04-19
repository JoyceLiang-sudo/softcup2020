# coding=utf-8
from easydict import EasyDict as edict

# yolo 的配置文件
conf = edict()

# 视频路径，为0则打开内置摄像头
conf.video_path = "./video.avi"
# conf.video_path = 0

# cfg 文件的路径
conf.cfg_path = './data/Gaussian_yolov3_BDD.cfg'

# weight文件的路径
conf.weight_path = './data/Gaussian_yolov3_BDD.weights'

# data文件的路径
conf.radar_data_path = './data/BDD.data'

# names文件路径
conf.names_path = './data/BDD.names'

# 置信度，大于此值会被判断为真
conf.thresh = 0.3
