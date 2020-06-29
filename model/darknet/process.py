# coding=utf-8
from model.darknet import darknet
from model.conf import conf


def darknet_process(pipe):
    """
    darknet进程
    """
    netMain = darknet.load_net_custom(conf.cfg_path.encode("ascii"), conf.weight_path.encode("ascii"), 0, 1)
    metaMain = darknet.load_meta(conf.radar_data_path.encode("ascii"))
    image_width = darknet.network_width(netMain)
    image_height = darknet.network_height(netMain)
    darknet_image = darknet.make_image(image_width, image_height, 3)
    pipe.send(image_width)
    pipe.send(image_height)
    while True:
        image = pipe.recv()
        darknet.copy_image_from_bytes(darknet_image, image.tobytes())

        # 类别编号  置信度 (x,y,w,h)
        detections = darknet.detect_image(netMain, metaMain, darknet_image,
                                          thresh=conf.thresh)
        pipe.send(detections)
