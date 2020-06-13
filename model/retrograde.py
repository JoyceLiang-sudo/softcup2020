from model.lane_line import get_intersection_point
from model.util.point_util import *


def make_range_lines(img, lane_lines):
    """
    制作检测区域
    """
    up_line = [[0, int(img.shape[0] / 4)], [img.shape[1], int(img.shape[0] / 4)]]
    down_line = [[0, img.shape[0]], [img.shape[1], img.shape[0]]]
    point = get_intersection_point(lane_lines[0], up_line)
    left_line = [point, lane_lines[0][1]]
    right_line = [[img.shape[1], 0], [img.shape[1], img.shape[0]]]
    range_lines = [up_line, down_line, left_line, right_line]
    return range_lines


def cal_x_y_weight(line_slope, car_point, lane_lines):
    """
    计算x,y分量，其中x为垂直坐标，y为平行于车道线的坐标
    """
    real_y = lane_lines[0][1][1] - car_point[1]
    real_x = int(real_y / line_slope)
    x_weight = car_point[0] - real_x - lane_lines[0][1][0]
    y_weight = np.sqrt(real_y * real_y + real_x * real_x)
    return x_weight, y_weight


def judge_run_direction(line_slope, track, lane_lines):
    """
    判断汽车行驶方向
    -1为下，0为停，1为上
    """
    if len(track) < 7:
        return 0
    x_weight1, y_weight1 = cal_x_y_weight(line_slope, track[-5], lane_lines)
    x_weight2, y_weight2 = cal_x_y_weight(line_slope, track[-1], lane_lines)
    if np.fabs(x_weight1 - x_weight2) > np.fabs(y_weight1 - y_weight2):
        return 0
    if y_weight2 > y_weight1:
        return 1
    if y_weight2 < y_weight1:
        return -1
    return 0


def get_retrograde_numbers(img, range_lines, tracks, line_slope, lane_lines):
    retrograde_numbers = []
    for track in tracks:
        if track[0] != 2:
            continue
        if track[-1][1] < range_lines[0][0][1]:
            continue

        if judge_point_line_position(track[-1], range_lines[2]) >= 0:
            continue
        flag = judge_run_direction(line_slope, track, lane_lines)
        if flag < 0:
            retrograde_numbers.append(track[1])

    return retrograde_numbers


def get_real_numbers(numbers, retrograde_numbers):
    """
    去除重复编号，返回真实编号集
    """
    for number in retrograde_numbers:
        flag = True
        for _number in numbers:
            if _number == number:
                flag = False
                break
        if flag:
            numbers.append(number)
    return numbers


def get_retrograde_cars(img, lane_lines, tracks, numbers):
    """
    得到逆行车辆
    """
    if len(lane_lines) <= 0:
        return 0
    range_lines = make_range_lines(img, lane_lines)
    # for line in range_lines:
    #     cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), [255, 200, 200], 5)
    line_slope = get_slope(lane_lines[0][0], lane_lines[0][1])
    retrograde_numbers = get_retrograde_numbers(img, range_lines, tracks, line_slope, lane_lines)
    real_numbers = get_real_numbers(numbers, retrograde_numbers)
    return real_numbers
