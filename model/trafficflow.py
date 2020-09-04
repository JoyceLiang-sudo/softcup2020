from model.util.point_util import *


class TrafficFlow:
    to_up_flow = 0
    to_down_flow = 0
    result_flow = 0
    pre_time = 0


# 计算当前帧的通过个数
def cal_traffic_count(reference_flow, tracks, track_kinds):
    to_up_flow = 0
    to_down_flow = 0
    for track in tracks:
        if track[0] != 13:
            continue
        if len(track) < track_kinds + 1:
            continue
        if judge_two_line_intersect(reference_flow[0], reference_flow[1], track[-1], track[-2]) or track[-1][1] == \
                reference_flow[0][1]:
            if track[-1][1] < track[-2][1]:
                to_up_flow = to_up_flow + 1
            else:
                to_down_flow = to_down_flow + 1
    return to_up_flow, to_down_flow


# 得到最终车流量
def get_result_flow(traffic_flow_class, to_up_flow, to_down_flow, time_up_flag):
    traffic_flow_class.to_up_flow = traffic_flow_class.to_up_flow + to_up_flow
    traffic_flow_class.to_down_flow = traffic_flow_class.to_down_flow + to_down_flow
    result_flow = traffic_flow_class.to_down_flow + traffic_flow_class.to_up_flow
    # 到一个周期更新一次车流量
    if time_up_flag:
        traffic_flow_class.result_flow = result_flow * 6
        traffic_flow_class.to_up_flow = 0
        traffic_flow_class.to_down_flow = 0
    return traffic_flow_class.result_flow


# 判断是否到时间周期
def judge_time_up(time, traffic_flow_class):
    if time - traffic_flow_class.pre_time > 10.0:
        traffic_flow_class.pre_time = time
        return True
    return False


def get_traffic_flow(img_width, img_height, traffic_flow_class, tracks, time, track_kinds):
    # 预置参考线
    reference_line = [[0, int(img_height / 2)], [img_width, int(img_height / 2)]]
    # 判断是否到时间周期
    time_up_flag = judge_time_up(time, traffic_flow_class)
    # 计算当前帧的通过个数
    to_up_flow, to_down_flow = cal_traffic_count(reference_line, tracks, track_kinds)
    # 得到最终车流量
    result_flow = get_result_flow(traffic_flow_class, to_up_flow, to_down_flow, time_up_flag)
    return result_flow
