import os
import shutil
import cv2
import time
from model.conf import conf


def save_in_file(img, box, file_name):
    roi_img = img[box[3][1]:box[4][1], box[3][0]:box[4][0]]
    frame_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    form = "%Y-%m-%d-%H_%M_%S"
    string_time = time.strftime(form, time.localtime(time.time()))
    print(string_time)
    cv2.imwrite(conf.save_path + file_name + "/" + string_time + ".jpg", frame_rgb)


def save_illegal_car(img, data, boxes):
    """
    保存违规车辆图片
    """
    for box in boxes:
        if box[0] != 2:
            continue
        for car in data.no_comity_pedestrian_cars_number:
            if box[5] == car:
                save_in_file(img, box, "no_comity_pedestrian")
                break
        for car in data.illegal_boxes_number:
            if box[5] == car:
                save_in_file(img, box, "illegal_change_lane")
                break
        for car in data.illegal_parking_numbers:
            if box[5] == car:
                save_in_file(img, box, "illegal_parking")
                break
        for car in data.true_running_car:
            if box[5] == car:
                save_in_file(img, box, "running_red")
                break
        for car in data.retrograde_cars_number:
            if box[5] == car:
                save_in_file(img, box, "retrograde_cars")
                break


def create_file(path):
    """
    创建新文件夹（工具函数）
    """
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def create_save_file():
    """
    创建所需的保存文件
    """
    folder = os.path.exists(conf.save_path)
    if folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        shutil.rmtree(conf.save_path)  # 如果存在则删除该文件夹
    create_file(conf.save_path)
    create_file(conf.save_path + conf.save_path1)
    create_file(conf.save_path + conf.save_path2)
    create_file(conf.save_path + conf.save_path3)
    create_file(conf.save_path + conf.save_path4)
    create_file(conf.save_path + conf.save_path5)
