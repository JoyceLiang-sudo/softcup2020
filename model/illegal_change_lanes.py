from model.util.point_util import *


def judge_illegal_change_lanes(img_height, tracks, lane_lines, illegal_boxes_number, track_kinds):
    """
    判断违规变道
    """
    illegal_cars = []
    for track in tracks:
        judge_flag = False
        if len(track) < track_kinds + 1:
            continue
        if track[0] != 2:
            continue
        for line in lane_lines:
            if judge_two_line_intersect(line[0], line[1], track[-1], track[-2]):
                illegal_cars.append(track[1])
                judge_flag = True
                break
        if judge_flag:
            continue
        if track[3][2] == 0:
            continue
        if track[3][3] != 0:
            continue
        judge_direction_wrong(img_height, track, illegal_cars)

    for number1 in illegal_cars:
        flag = True
        for number2 in illegal_boxes_number:
            if number1 == number2:
                flag = False
                break
        if flag:
            illegal_boxes_number.append(number1)
    return illegal_boxes_number


def judge_direction_wrong(img_height, track, illegal_cars):
    left_flag = judge_point_line_position(track[-1], track[3][0]) * judge_point_line_position(track[-2],
                                                                                              track[3][0]) <= 0
    right_flag = judge_point_line_position(track[-1], track[3][1]) * judge_point_line_position(track[-2],
                                                                                               track[3][1]) <= 0
    out_range_flag = track[-1][1] < int(img_height / 2)
    # 如果方向为左转
    if track[3][2] == 1:
        if left_flag:
            track[3][3] = 1
            return False
        if out_range_flag:
            track[3][3] = -1
            return True
        return False

    # 如果方向为直行
    if track[3][2] == 2:
        if left_flag:
            illegal_cars.append(track[1])
            track[3][3] = -1
            return True
        if right_flag:
            illegal_cars.append(track[1])
            track[3][3] = -1
            return True
        if out_range_flag:
            track[3][3] = 1
            return False
        return False
    # 如果方向为右转
    if track[3][2] == 3:
        if right_flag:
            track[3][3] = 1
            return False
        if out_range_flag:
            track[3][3] = -1
            return True
        return False
    # 如果方向为左转加直行
    if track[3][2] == 12:
        if right_flag:
            illegal_cars.append(track[1])
            track[3][3] = -1
            return True
        if out_range_flag:
            track[3][3] = 1
            return False
        return False
    # 如果方向为右转加直行
    if track[3][2] == 23:
        if left_flag:
            illegal_cars.append(track[1])
            return True
        if out_range_flag:
            track[3][3] = 1
            return False
        return False