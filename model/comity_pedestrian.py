from model.lane_line import get_intersection_point
from model.util.point_util import *


class Comity_Pedestrian(object):
    car_pass_cars_people = []
    person_pass_cars_people = []


# 解出行人前沿轨迹
def get_predict_people_lines(img, tracks):
    predict_people_lines = []
    car_tracks = []
    people_tracks = []
    for track in tracks:
        if track[0] == 2:
            car_tracks.append(track)
            continue
        if track[0] != 4:
            continue
        people_tracks.append(track)
        predict_people_line = get_predict_people_line(img, track)
        predict_people_lines.append(predict_people_line)
    return predict_people_lines, car_tracks, people_tracks


# 解出单个行人前沿轨迹
def get_predict_people_line(img, track):
    if len(track) < 3:
        predict_people_line = [[0, 0], [0, 0]]
        return predict_people_line
    long_line = [[track[-1][0], track[-1][1]], [track[2][0], track[2][1]]]
    slope = get_slope(track[-1], track[2])
    possible_points = get_possible_points(img, slope, long_line)
    real_point = get_another_point(track, possible_points)
    predict_people_line = [track[1], track[-1], real_point]
    return predict_people_line


# 解出行人前沿轨迹的图像边界点
def get_another_point(track, possible_points):
    pre_person_point = track[2]
    now_person_point = track[-1]
    if pre_person_point[0] == now_person_point[0]:
        pre_person_point = [pre_person_point[0] + 1, pre_person_point[1]]
    for possible_point in possible_points:
        if (possible_point[0] - now_person_point[0]) * (now_person_point[0] - pre_person_point[0]) > 0:
            return possible_point

    return [0, 0]


# 解出行人轨迹直线的可能图像边界点
def get_possible_points(img, slope, long_line):
    line1, line2 = get_another_lines(img, slope, long_line)
    point1 = get_intersection_point(long_line, line1)
    point2 = get_intersection_point(long_line, line2)
    possible_points = [point1, point2]
    return possible_points


# 解出对应的图像边界线
def get_another_lines(img, slope, long_line):
    width = img.shape[1]
    height = img.shape[0]
    left_line = [[0, 0], [1, height]]
    right_line = [[width - 1, 0], [width, height]]
    up_line = [[0, 0], [width, 0]]
    down_line = [[0, height], [width, height]]
    if slope == 0:
        return [left_line, right_line]
    point_left = get_intersection_point(long_line, left_line)
    point_right = get_intersection_point(long_line, right_line)
    point_up = get_intersection_point(long_line, up_line)
    point_down = get_intersection_point(long_line, down_line)
    if 0 < point_left[1] < height:
        if 0 < point_up[0] < width:
            return left_line, up_line
        if 0 < point_right[1] < height:
            return left_line, right_line
        if 0 < point_down[0] < width:
            return left_line, down_line
    elif 0 < point_up[0] < width:
        if 0 < point_right[1] < height:
            return up_line, right_line
        if 0 < point_down[0] < width:
            return up_line, down_line
    return right_line, down_line


# 当车压过人的前沿轨迹时，找到对应车和人
def find_car_pass(predict_people_lines, car_tracks, comity_pedestrian):
    car_pass_cars_people = []
    for car_track in car_tracks:
        for predict_people_line in predict_people_lines:
            if judge_two_line_intersect(predict_people_line[1], predict_people_line[2], car_track[2], car_track[-1]):
                car_pass_cars_people.append([car_track[1], car_track[-1],
                                             predict_people_line[0], predict_people_line[1]])
    for car_pass_car_person in car_pass_cars_people:
        flag = True
        for real_car_pass_car_person in comity_pedestrian.car_pass_cars_people:
            if car_pass_car_person[0] == real_car_pass_car_person[0] and car_pass_car_person[2] == \
                    real_car_pass_car_person[2]:
                flag = False
                break
        if flag:
            comity_pedestrian.car_pass_cars_people.append(car_pass_car_person)


# 当人走过车的行驶轨迹时，找到对应车和人
def find_people_pass(people_tracks, car_tracks, comity_pedestrian):
    person_pass_cars_people = []
    for car_track in car_tracks:
        for people_track in people_tracks:
            if judge_two_line_intersect(car_track[2], car_track[-1], people_track[2], people_track[-1]):
                person_pass_cars_people.append([car_track[1], car_track[-1],
                                                people_track[1], people_track[-1]])
    for person_pass_car_person in person_pass_cars_people:
        flag = True
        for real_person_pass_car_person in comity_pedestrian.person_pass_cars_people:
            if person_pass_car_person[0] == real_person_pass_car_person[0] and person_pass_car_person[2] == \
                    real_person_pass_car_person[2]:
                flag = False
                break
        if flag:
            comity_pedestrian.person_pass_cars_people.append(person_pass_car_person)


# 得到最后结构
def get_result_cars_people(comity_pedestrian):
    result_cars_people = []
    for car_pass_car_person in comity_pedestrian.car_pass_cars_people:
        for person_pass_car_person in comity_pedestrian.person_pass_cars_people:
            if car_pass_car_person[0] == person_pass_car_person[0] and car_pass_car_person[2] == person_pass_car_person[
                2]:
                result_cars_people.append(car_pass_car_person[0])

    return result_cars_people


def judge_comity_pedestrian(img, tracks, comity_pedestrian, numbers):
    # 解出行人前沿轨迹
    predict_people_lines, car_tracks, people_tracks = get_predict_people_lines(img, tracks)
    # 当车压过人的前沿轨迹时，找到对应车和人
    find_car_pass(predict_people_lines, car_tracks, comity_pedestrian)
    # 当人走过车的行驶轨迹时，找到对应车和人
    find_people_pass(people_tracks, car_tracks, comity_pedestrian)
    # 得到最后结构
    result_cars_people = get_result_cars_people(comity_pedestrian)
    # 取消重复项
    result_cars_people = find_real_numbers(numbers, result_cars_people)
    return result_cars_people
