import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
import cv2


def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_stop_line(img, stop_line, color=[200, 255, 0], thickness=2):
    cv2.line(img, (stop_line[0][0], stop_line[0][1]), (stop_line[1][0], stop_line[1][1]), color, thickness)


def draw_lane_lines(img, lane_lines, color=[255, 0, 0], thickness=2):
    """
    画车道线
    """
    for line in lane_lines:
        cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color, thickness)


def find_lines(lines):
    lane_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 1 or slope < -0.5:
                line_one = [[x1, y1], [x2, y2]]
                lane_lines.append(line_one)
    return lane_lines


def extend_lines(img, zebra_crossing, lane_lines, points):
    up_points = []
    bottom_points = []
    line_bottom = [[0, img.shape[0]], [img.shape[1], img.shape[0]]]
    line_left = [[0, 0], [1, img.shape[0]]]
    for line in lane_lines:
        point = get_intersection_point(zebra_crossing, line)
        up_points.append(point)
        point = get_intersection_point(line_bottom, line)
        if point[0] < 0:
            point = get_intersection_point(line_left, line)
        bottom_points.append(point)
    points.append(up_points)
    points.append(bottom_points)
    resize_point(points)
    lane_lines.clear()
    i = 0
    while i < len(points[0]):
        # points[0]、points[1]分别为上下点集
        line_one = [[points[0][i][0], points[0][i][1]], [points[1][i][0], points[1][i][1]]]
        lane_lines.append(line_one)
        i = i + 1


def resize_point(points):
    for point in points:
        for _point in point:
            _point[0] = _point[0] * 2
            _point[1] = _point[1] * 2


def get_intersection_point(line1, line2):
    if line1[0][0] == line1[1][0]:
        line1[0][0] = line1[0][0] + 1
    if line2[0][0] == line2[1][0]:
        line2[0][0] = line2[0][0] + 1
    point = []
    k1 = (line1[0][1] - line1[1][1]) / (line1[0][0] - line1[1][0])
    b1 = line1[0][1] - k1 * line1[0][0]
    k2 = (line2[0][1] - line2[1][1]) / (line2[0][0] - line2[1][0])
    b2 = line2[0][1] - k2 * line2[0][0]
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    point.append(int(x))
    point.append(int(y))
    return point


def find_left_line(lane_lines):
    if len(lane_lines) <= 0:
        return 0
    left_x = lane_lines[0][0][0]
    left_line = lane_lines[0]
    for line in lane_lines:
        if left_x > line[0][0]:
            left_x = line[0][0]
            left_line = line
    return left_line


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, zebra_crossing, points):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lane_lines = find_lines(lines)
    extend_lines(line_img, zebra_crossing, lane_lines, points)
    return lane_lines


def deal_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        rect = cv2.minAreaRect(contours[0])
        x, y = rect[0]
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), 5)
        cv2.drawContours(img, contour, -1, (255, 255, 0), 3)
    return img


def get_stop_line(zebra_width, zebra_crossing, left_line):
    stop_line = [[zebra_crossing[0][0], int(zebra_crossing[0][1] * 2 + zebra_width * 0.7)],
                 [zebra_crossing[1][0], int(zebra_crossing[1][1] * 2 + zebra_width * 0.7)]]
    left_point = get_intersection_point(stop_line, left_line)
    real_stop_line = [left_point, [zebra_crossing[1][0], int(zebra_crossing[1][1] * 2 + zebra_width * 0.7)]]
    return real_stop_line


def deal_picture(img, zebra_crossing, points, zebra_width):
    blur_ksize = 5
    canny_lthreshold = 100
    canny_hthreshold = 150
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 123
    max_line_gap = 20
    x, y = img.shape[0:2]
    img = cv2.resize(img, (int(y / 2), int(x / 2)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = deal_contours(gray)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_vtx = np.array([[(0, img.shape[0]), (20, 325), (img.shape[1] - 50, 325), (img.shape[1], img.shape[0])]])
    roi_edges = roi_mask(edges, roi_vtx)
    lane_lines = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap, zebra_crossing
                             , points)
    # 车道线排序
    lane_lines.sort()
    stop_line = get_stop_line(zebra_width, zebra_crossing, lane_lines[0])
    return lane_lines, stop_line


def get_lane_lines(img, zebra_line):
    points = []
    zebra_width = zebra_line.ymax - zebra_line.ymin
    point1 = [0, (zebra_line.ymax + zebra_line.ymin) / 4]
    point2 = [img.shape[1], (zebra_line.ymax + zebra_line.ymin) / 4]
    zebra_crossing = [point1, point2]
    lane_lines, stop_line = deal_picture(img, zebra_crossing, points, zebra_width)
    return lane_lines, stop_line


def cal_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
