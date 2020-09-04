from model.util.point_util import *


def judge_illegal_change_lanes(tracks, lane_lines, illegal_boxes_number, track_kinds):
    """
    判断违规变道
    """
    if lane_lines is None:
        return illegal_boxes_number
    illegal_cars = []
    for track in tracks:
        judge_flag = False
        if len(track) < track_kinds + 1:
            continue
        if track[0] != 19:
            continue
        if track[-1][1] > 1600:
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

    for number1 in illegal_cars:
        flag = True
        for number2 in illegal_boxes_number:
            if number1 == number2:
                flag = False
                break
        if flag:
            illegal_boxes_number.append(number1)
    return illegal_boxes_number


def judge_person_illegal_through_road(tracks, zebra_crossing, img_width, illegal_number):
    illegal_person = []
    for track in tracks:
        if track[-1][0] < int(img_width * 1 / 3):
            continue
        if track[-1][0] > int(img_width * 2 / 3):
            continue
        if track[0] != 10:
            continue
        if zebra_crossing is None:
            illegal_person.append(track[1])
    for number1 in illegal_person:
        flag = True
        for number2 in illegal_number:
            if number1 == number2:
                flag = False
                break
        if flag:
            illegal_number.append(number1)
    return illegal_number


def judge_drive_wrong_direction(img_height, tracks, illegal_cars, track_kinds):
    drive_wrong_direction = []
    for track in tracks:
        if len(track) < track_kinds + 1:
            continue
        if track[0] != 19:
            continue
        if not judge_direction_wrong(img_height, track, illegal_cars):
            continue
        drive_wrong_direction.append(track[1])
    for number1 in drive_wrong_direction:
        flag = True
        for number2 in illegal_cars:
            if number1 == number2:
                flag = False
                break
        if flag:
            illegal_cars.append(number1)
    return illegal_cars


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
