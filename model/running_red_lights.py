from model.util.point_util import judge_two_line_intersect


def judge_running_car(boxes, running_car, tracks, zebra, track_kinds):
    """
    检测闯红灯
    running_car[0]为 可能 闯红灯车追踪编号
    running_car[1]为 确定 闯红灯车追踪编号
    """

    if zebra.up_zebra_line is None or zebra.down_zebra_line is None:
        # 先确定有两条斑马线
        return

    red_light_flag = False
    for box in boxes:
        if box[0] not in [4, 8, 12]:
            red_light_flag = True
    if not red_light_flag:
        # 没有识别到红灯
        return

    for track in tracks:

        if track[0] in [2, 6, 13] and len(track) > track_kinds + 1:

            if track[1] in running_car[1]:
                # 该车辆已经判定为闯红灯
                continue

            if track[1] in running_car[0]:
                # 该车辆已经在可疑列表中
                if judge_two_line_intersect(zebra.up_zebra_line[0], zebra.up_zebra_line[1], track[-1], track[4]):
                    # 轨迹穿过上斑马线，判定为闯红灯
                    running_car[1].append(track[1])
                continue

            if judge_two_line_intersect(zebra.down_zebra_line[0], zebra.down_zebra_line[1], track[-1], track[4]):
                # 车轨迹穿过下斑马线，加入可疑列表
                running_car[0].append(track[1])
