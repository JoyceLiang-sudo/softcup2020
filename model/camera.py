from model.util.point_util import *


def set_camera_message():
    """
    设置相机参数（以十字路口右下角点为原点，→x，↓y，高z，单位cm）
    (x,y,z)
    :return:相机（场地）信息数组
    """
    camera_message = []
    x = 100
    y = 100
    z = 1000
    camera_message.append(x)
    camera_message.append(y)
    camera_message.append(z)
    return camera_message
