from model.util.point_util import *
from model.lane_line import deal_contours, roi_mask
from model.conf import conf
import sys


def get_possible_area(lanes):
    """
    得到疑似区域
    """
    return lanes[-1]


def find_lines(lines):
    """
    找目标线
    """
    real_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + sys.float_info.min)
            if 0.1 < slope < 2:
                line_one = [[x1, y1], [x2, y2]]
                real_lines.append(line_one)
    return real_lines


def get_roi_mat(img, points):
    """
    得到roi
    """
    blur_k_size = 5
    canny_l_threshold = 100
    canny_h_threshold = 150
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = deal_contours(gray)
    blur_gray = cv2.GaussianBlur(gray, (blur_k_size, blur_k_size), 0, 0)
    edges = cv2.Canny(blur_gray, canny_l_threshold, canny_h_threshold)
    roi_vtx = np.array([[(points[0][0], points[0][1]), (points[1][0], points[1][1]), (points[2][0], points[2][1]),
                         (points[3][0], points[3][1])]])
    roi_edges = roi_mask(edges, roi_vtx)
    return roi_edges


def get_judge_result(lines):
    """
    得到最终的判断
    """
    if len(lines) <= 1:
        return False
    angles = []
    angles_sub = []
    for line in lines:
        slope = (line[0][1] - line[1][1]) / (line[0][0] - line[1][0])
        angles.append(np.arctan(slope) * 180.0 / np.pi)
    flag = False
    pre_angle = angles[0]
    for angle in angles:
        if not flag:
            flag = True
            continue
        now_angle = angle
        angle_sub = now_angle - pre_angle
        if np.fabs(angle_sub) < 5:
            angles_sub.append(angle_sub)
    if len(angles_sub) < 5:
        return False
    return True


def get_roi_points(img, possible_area, stop_line):
    """
    确定roi的四个点
    """
    down_line = [[0, img.shape[0]], [img.shape[1], img.shape[0]]]
    point1 = get_intersection_point(possible_area[0], stop_line)
    point2 = get_intersection_point(possible_area[1], stop_line)
    point3 = get_intersection_point(possible_area[0], down_line)
    point4 = get_intersection_point(possible_area[1], down_line)
    points = [point3, point1, point2, point4]
    return points


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    霍夫变换找目标线
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    real_lines = find_lines(lines)
    return real_lines, line_img


def judge_illegal_area(img, lanes, stop_line):
    """
    判断是否有违停区域
    """
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 70
    max_line_gap = 20
    possible_area = get_possible_area(lanes)
    points = get_roi_points(img, possible_area, stop_line)
    roi_mat = get_roi_mat(img, points)
    lines, line_img = hough_lines(roi_mat, rho, theta, threshold, min_line_length, max_line_gap)
    flag = get_judge_result(lines)
    real_area = [possible_area, stop_line]
    return flag, real_area


def find_illegal_area(img, lanes, stop_line):
    """
    找出违停区域(容量为2,下标为0的为左右两条线集合，下标为1的为停车线)
    """
    flag, possible_area = judge_illegal_area(img, lanes, stop_line)
    if flag:
        illegal_area = possible_area
    else:
        illegal_area = []
    return illegal_area


def judge_car_parking(track):
    """
    判断车是否停下
    """
    count = 0
    for t in track:
        if count < 2:
            count = count + 1
            continue


def find_illegal_area_cars(illegal_area, tracks, track_kinds):
    """
    找出在违停区域的车
    """
    if len(illegal_area) <= 0:
        return []
    illegal_numbers = []
    for track in tracks:
        if track[1] != 2:
            continue
        if len(track) < 4 + track_kinds:
            continue
        if judge_point_line_position(track[-1], illegal_area[0][0]) != 1:
            continue
        if judge_point_line_position(track[-1], illegal_area[0][1]) != -1:
            continue
        if track[-1][1] <= illegal_area[1][0][1]:
            continue
        if calculate_average_deviation([track[-1], track[-2], track[-3], track[-4], track[-5]]) > 10:
            continue
        illegal_numbers.append(track[0])
    return illegal_numbers


def find_illegal_parking_cars(illegal_area, tracks, numbers, track_kinds):
    """
    找出在违停区域停车的车
    """
    illegal_cars = find_illegal_area_cars(illegal_area, tracks, track_kinds)
    now_numbers = find_real_numbers(numbers, illegal_cars)
    return now_numbers
