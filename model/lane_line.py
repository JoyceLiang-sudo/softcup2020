from model.util.point_util import *


def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_stop_line(img, stop_line, color=(0, 215, 255), thickness=10):
    if len(stop_line) > 0:
        cv2.line(img, (stop_line[0][0], stop_line[0][1]), (stop_line[1][0], stop_line[1][1]), color, thickness)


def draw_lane_lines(img, lane_lines, color=(0, 0, 255), thickness=16):
    """
    画车道线
    """
    if lane_lines is not None:
        for line in lane_lines:
            cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color, thickness)


def find_lines(lines):
    if lines is None:
        return []
    lane_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0.8 or slope < -1:
                line_one = [[x1, y1], [x2, y2]]
                lane_lines.append(line_one)
    return lane_lines


def extend_lines(img, zebra_crossing, lane_lines):
    if zebra_crossing[0][1] == 0:
        # reference_line = [[0, int(img.shape[0] / 5 * 3)], [img.shape[1], int(img.shape[0] / 5 * 3)]]
        reference_line = [[0, 100], [img.shape[1], 100]]
    else:
        reference_line = zebra_crossing
    # reference_line = [[0, int(img.shape[0] / 5 * 2)], [img.shape[1], int(img.shape[0] / 5 * 2)]]
    points = []
    up_points = []
    bottom_points = []
    line_bottom = [[0, img.shape[0]], [img.shape[1], img.shape[0]]]
    line_left = [[0, 0], [1, img.shape[0]]]
    for line in lane_lines:
        point = get_intersection_point(reference_line, line)
        # point = line[0]
        up_points.append(point)
        point = get_intersection_point(line_bottom, line)
        # point = line[1]
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


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, zebra_crossing):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lane_lines = find_lines(lines)
    lane_lines = remove_short_lines(lane_lines)
    if zebra_crossing[0][1] == 0:
        lane_lines = [find_longest_line(lane_lines)]
    extend_lines(line_img, zebra_crossing, lane_lines)

    return lane_lines


def deal_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        cv2.drawContours(img, contour, -1, (255, 255, 0), 3)
    return img


def get_stop_line(img, zebra_width, zebra_crossing, lane_lines):
    if len(lane_lines) <= 0:
        left_line = [[0, 0], [0, img.shape[0]]]
    else:
        left_line = lane_lines[0]
    stop_line = [[zebra_crossing[0][0], int(zebra_crossing[0][1] * 2 + zebra_width * 0.7)],
                 [zebra_crossing[1][0], int(zebra_crossing[1][1] * 2 + zebra_width * 0.7)]]
    left_point = get_intersection_point(stop_line, left_line)
    real_stop_line = [left_point, [zebra_crossing[1][0], int(zebra_crossing[1][1] * 2 + zebra_width * 0.7)]]
    return real_stop_line


def deal_picture(img, zebra_crossing, zebra_width, template_img, init_flag, boxes):
    blur_k_size = 5
    canny_l_threshold = 80  # 80
    canny_h_threshold = 150
    rho = 1
    theta = np.pi / 180
    threshold = 20
    min_line_length = 130
    max_line_gap = 40
    x, y = img.shape[0:2]
    img = cv2.resize(img, (int(y / 2), int(x / 2)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retval, gray_test = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    gray = deal_contours(gray_test)
    blur_gray = cv2.GaussianBlur(gray, (blur_k_size, blur_k_size), 0, 0)
    edges = cv2.Canny(blur_gray, canny_l_threshold, canny_h_threshold)
    roi_vtx = np.array([[(0, img.shape[0]), (0, int(img.shape[0] / 2)), (img.shape[1], int(img.shape[0] / 2)),
                         (img.shape[1], img.shape[0])]])
    roi_edges = roi_mask(edges, roi_vtx)
    if init_flag:
        corners_message = find_line_contours(img, template_img, boxes)
    lane_lines = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap, zebra_crossing)
    # 车道线排序
    lane_lines.sort()

    stop_line = get_stop_line(img, zebra_width, zebra_crossing, lane_lines)

    if init_flag:
        return lane_lines, stop_line, corners_message
    return lane_lines, stop_line


def find_line_contours(img_pre, template_imgs_list, boxes):
    length = 0
    for box in boxes:
        if box[0] not in [0, 1, 5]:
            continue
        if box[2][1] < 1000:
            continue
        length += box[0]
    if length == 6:  # 2
        template_imgs = template_imgs_list[2]
    elif length == 5:  # 6
        template_imgs = template_imgs_list[0]
    elif length == 1:  # 8
        template_imgs = template_imgs_list[1]
    else:
        return []
    corners_message = []
    for template_img in template_imgs:
        tl, br = template_demo(template_img, img_pre)
        tl = [tl[0] * 2, tl[1] * 2]
        br = [br[0] * 2, br[1] * 2]
        corner_message = [tl, br]
        corners_message.append(corner_message)
    return corners_message


def get_lane_lines(img, zebra_line, template_img, init_flag, boxes):
    """
    获得车道线
    :param img: 输入图像
    :param zebra_line: 斑马线
    :param template_img: 模板
    :param init_flag: 初始化标志位
    :param boxes: boxes
    :return: 车道线
    """
    for box in boxes:
        if box[0] not in [0, 1, 5]:
            continue
        print(box)

    find_src = img.copy()
    if zebra_line is None:
        zebra_width = img.shape[1]
        zebra_line = [[0, 0], [0, 0], [0, 0], [0, 0]]
    else:
        zebra_width = zebra_line[3][1] - zebra_line[2][1]
    point1 = [0, (zebra_line[3][1] + zebra_line[2][1]) / 4]
    point2 = [img.shape[1], (zebra_line[3][1] + zebra_line[2][1]) / 4]
    zebra_crossing = [point1, point2]
    if init_flag:
        result_lines, stop_line, corners_message = deal_picture(img, zebra_crossing, zebra_width, template_img,
                                                                init_flag, boxes)
        result_lines = find_result_lane_lines(find_src, corners_message, result_lines, zebra_line)
        return result_lines, stop_line
    result_lines, stop_line = deal_picture(img, zebra_crossing, zebra_width, template_img, init_flag, boxes)
    return result_lines


def cal_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def make_lines_group(lane_lines):
    """
    把所有车道线分组
    """
    lines_group = []
    for line1 in lane_lines:
        flag = True
        lines = []
        for line_group in lines_group:
            for line_group1 in line_group:
                if line1[0][0] == line_group1[0][0]:
                    flag = False
                    break
            if not flag:
                break
        if not flag:
            continue
        for line2 in lane_lines:
            if np.fabs(line1[0][0] - line2[0][0]) < 200:
                lines.append(line2)
        lines_group.append(lines)
    return lines_group


def make_lanes(img, lines_group):
    """
    组成车道
    """
    lanes = []
    if len(lines_group) == 1:
        lanes.append([[lines_group[0][-1]], [[img.shape[1], 0], [img.shape[1], img.shape[0]]]])
        return lanes
    left_line = lines_group[0]
    flag = False
    for i in range(0, len(lines_group)):
        if not flag:
            flag = True
            continue
        right_line = lines_group[i]
        lanes.append([left_line, right_line])
        left_line = lines_group[i]
        if i == len(lines_group) - 1:
            if right_line[1][1] == img.shape[0]:
                right_line = [[img.shape[1], 0], [img.shape[1], img.shape[0]]]
                lanes.append([left_line, right_line])
    return lanes


def get_lanes(img, lane_lines, init_flag):
    """
    得到车道（每个元素包含左线，右线）
    """
    if len(lane_lines) > 0:
        lines_group = make_lines_group(lane_lines)
        if init_flag:
            lines_group = union_lane_lines(lines_group)
        lanes = make_lanes(img, lines_group)
        if init_flag:
            return lanes, lines_group
        return lanes, []
    return [], []


def set_lanes_message(boxes, lanes):
    """
    输入车道信息
    0-无，1-左转，2-直行，3-右转，12-左转加直行，23-右转加直行，4-公交专用道，5-禁止停车道
    :return:车道信息数组
    """
    if lanes is None:
        return []
    if len(lanes) == 0:
        return []
    if len(lanes) == 1:
        return [0]
    lane_message = []
    for i in range(0, len(lanes)):
        lane_message.append(0)
    for box in boxes:
        if box[0] != 0 and box[0] != 1 and box[0] != 5:
            continue
        if box[2][1] < lanes[0][0][0][1]:
            continue
        for i in range(0, len(lanes)):
            if judge_point_in_lines(box[2], lanes[i][0], lanes[i][1]):
                if box[0] == 0:
                    lane_message[i] = 2
                if box[0] == 1:
                    lane_message[i] = 1
                if box[0] == 5:
                    lane_message[i] = 3
                break
    if lane_message[0] == 0:
        lane_message[0] = 1
    return lane_message


def make_tracks_lane(tracks, lanes, stop_line, lanes_message):
    """
    为每个车辆轨迹匹配车道
    :param tracks: 车辆轨迹数组
    :param lanes: 车道
    :param stop_line: 停车线
    :param lanes_message: 车道信息
    :return: tracks
    """
    now_tracks = []
    message_vector = []
    for track in tracks:
        t_track = track
        message = []
        lane_message = []
        # 如果类别不是车辆，则过滤
        if track[0] != 13:
            now_tracks.append(t_track)
            continue
        # 如果车辆在停车线上方，则过滤
        if track[-1][1] < stop_line[0][1]:
            now_tracks.append(t_track)
            continue
        # 如果车辆在最左侧车道线左边，则过滤
        if judge_point_line_position(track[-1], lanes[0][0]) != -1:
            now_tracks.append(t_track)
            continue
        # 如果车辆在最右侧车道线右边，则过滤
        if judge_point_line_position(track[-1], lanes[-1][1]) != 1:
            now_tracks.append(t_track)
            continue
        # 如果车辆已被分配有向车道，则过滤
        if track[3][2] != 0:
            now_tracks.append(t_track)
            continue
        for i in range(0, len(lanes)):
            if judge_point_in_lines(track[-1], lanes[i][0], lanes[i][1]):
                message.append(track[0])
                message.append(track[1])
                message.append(track[2])
                lane_message.append(lanes[i][0])
                lane_message.append(lanes[i][1])
                lane_message.append(lanes_message[i])
                lane_message.append(track[3][3])
                message.append(lane_message)
                for j in range(4, len(track)):
                    message.append(track[j])
                message_vector.append(message)
                break
    now_tracks.extend(message_vector)
    return now_tracks


def find_result_lane_lines(img, corners_message, lane_lines, zebra_lines):
    """
    筛出不符合位置条件的车道线，得到最终结果
    :param corners_message: 车道角点信息
    :param lane_lines: 原车道线
    :return: 最终车道线
    """
    result_lines = []
    result_line = lane_lines[0]
    max_length = 0
    for line in lane_lines:
        length = calculate_two_point_distance(line[0][0], line[0][1], line[1][0], line[1][1])
        if length > max_length:
            max_length = length
            result_line = line
        if len(corners_message) < 2:
            continue
        for corner_message in corners_message:
            line_top = [[corner_message[0][0], corner_message[0][1]], [corner_message[1][0], corner_message[0][1]]]
            line_bottom = [[corner_message[0][0], corner_message[1][1]], [corner_message[1][0], corner_message[1][1]]]
            flag1 = judge_two_line_intersect(line_top[0], line_top[1], line[0], line[1])
            if line_bottom[1][1] > img.shape[0]:
                flag2 = False
            else:
                flag2 = judge_two_line_intersect(line_bottom[0], line_bottom[1], line[0], line[1])
            if flag1 or flag2:
                result_lines.append(line)
                break
    if zebra_lines[1][0] == 0:
        return [result_line]
    return result_lines


def remove_short_lines(lane_lines):
    """
    去除较短的车道线
    :param lane_lines: 车道线
    :return: 结果车道线
    """
    result_lines = []
    max_length = 0
    for line in lane_lines:
        length = calculate_two_point_distance(line[0][0], line[0][1], line[1][0], line[1][1])
        if length > max_length:
            max_length = length
    for line in lane_lines:
        length = calculate_two_point_distance(line[0][0], line[0][1], line[1][0], line[1][1])
        if length > max_length * 0.1:
            result_lines.append(line)
    return lane_lines


def make_adjoin_matching(pre_lane_lines, now_lane_lines):
    """
    进行更新车道线匹配
    :param pre_lane_lines: 之前的车道线
    :param now_lane_lines: 现在的车道线
    :return: 最终的车道线
    """
    result_lines = []
    for now_lane_line in now_lane_lines:
        for pre_lane_line in pre_lane_lines:
            if np.fabs(now_lane_line[0][0] - pre_lane_line[0][0]) < 50 and np.fabs(
                    now_lane_line[1][0] - pre_lane_line[1][0]) < 50:
                result_lines.append(now_lane_line)
                break
    return result_lines


def protect_lanes(pre_lane_lines, now_lane_lines, pre_lanes, now_lanes):
    """
    保护车道
    :param now_lane_lines: 之前的车道线
    :param pre_lane_lines: 现在的车道线
    :param pre_lanes: 之前的车道
    :param now_lanes: 现在的车道
    :return: 最终车道线，最终车道
    """
    if len(pre_lanes) != len(now_lanes):
        return pre_lane_lines, pre_lanes
    if np.fabs(pre_lanes[-1][0][0][0] - now_lanes[-1][0][0][0]) > 100:
        return pre_lane_lines, pre_lanes
    return now_lane_lines, now_lanes


def union_lane_lines(lines_group):
    result_lines = []
    for line_group in lines_group:
        if len(line_group) == 1:
            result_lines.append(line_group[0])
        elif len(line_group) == 2:
            point1 = [int((line_group[0][0][0] + line_group[1][0][0]) / 2),
                      int((line_group[0][0][1] + line_group[1][0][1]) / 2)]
            point2 = [int((line_group[0][1][0] + line_group[1][1][0]) / 2),
                      int((line_group[0][1][1] + line_group[1][1][1]) / 2)]
            result_lines.append([point1, point2])
        else:
            now_line_group = remove_wrong_slope_lines(line_group)
            if len(now_line_group) == 0:
                now_line_group = line_group
            point1, point2 = union_one_group(now_line_group)
            result_lines.append([point1, point2])
    return result_lines


def union_one_group(line_group):
    if len(line_group) == 0:
        return [0, 0], [0, 0]
    sum_x1 = 0
    sum_x2 = 0
    for line in line_group:
        sum_x1 += line[0][0]
        sum_x2 += line[1][0]
    point1 = line_group[int(len(line_group) / 2)][0]
    point2 = line_group[int(len(line_group) / 2)][1]
    return point1, point2


def remove_wrong_slope_lines(line_group):
    result_lines = []
    min_slope = 100
    max_slope = 0
    line_group_structs = []
    for line in line_group:
        slope = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        if slope > max_slope:
            max_slope = slope
        if slope < min_slope:
            min_slope = slope
        line_group_struct = [line, slope]
        line_group_structs.append(line_group_struct)
    for line_group_struct in line_group_structs:
        if line_group_struct[1] != min_slope and line_group_struct[1] != max_slope:
            result_lines.append(line_group_struct[0])
    return result_lines


def find_lane_lines_position_range(lane_lines, img_width):
    """
    找到车道线位置范围
    :param lane_lines: 车道线
    :param img_width: 图像宽度
    :return: 位置范围
    """
    if lane_lines is None:
        return [], []
    position_range = []
    if lane_lines[0][0][0] >= 200:
        range_left = lane_lines[0][0][0] - 200
    else:
        range_left = 0
    spaces = []
    for i in range(0, len(lane_lines)):
        if lane_lines[i] == lane_lines[-1]:
            range_right = img_width
            position_range.append([range_left, range_right])
            break
        spaces.append(
            [lane_lines[i + 1][0][0] - lane_lines[i][0][0], lane_lines[i + 1][1][0] - lane_lines[i][1][0]])
        range_right = lane_lines[i][0][0] + int((lane_lines[i + 1][0][0] - lane_lines[i][0][0]) / 2)
        position_range.append([range_left, range_right])
        range_left = range_right
    return position_range, spaces


def find_longest_line(lane_lines):
    if lane_lines is None:
        return None
    max_length = 0
    result_line = lane_lines[0]
    for line in lane_lines:
        length = calculate_two_point_distance(line[0][0], line[0][1], line[1][0], line[1][1])
        if length > max_length:
            max_length = length
            result_line = line
    return result_line
