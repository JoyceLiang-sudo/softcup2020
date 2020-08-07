def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def judge_running_car(frame_read, origin, running_car, boxes, tracks, stop_line, lane_lines, track_kinds):
    true_running_car = []
    if boxes:
        for box in boxes:
            if box[6] == 'red' and lane_lines[0][0][0] < box[3][0] < lane_lines[-1][0][0]:
                for track in tracks:
                    # 2为car
                    if track[0] == 2:
                        # 若之前闯过，继续跟踪
                        if track[1] in running_car:
                            if track[-1] == track[-2]:
                                if not track[1] in origin:
                                    origin.append(track[1])
                                if track[-1][1] > frame_read.shape[0]:
                                    running_car.remove(track[1])
                        else:
                            if len(track) > track_kinds + 1:
                                for i in range(0, len(origin)):
                                    if track[1] == origin[i]:
                                        if track[-1] != track[-2]:
                                            if not track[1] in running_car:
                                                running_car.append(track[1])
                                if intersect((stop_line[0][0], stop_line[0][1]), (stop_line[1][0], stop_line[1][1]),
                                         track[-1],
                                         track[-2]):
                                    running_car.append(track[1])
                        if track[1] in running_car:
                            if track[-1][1] < frame_read.shape[0]/3:
                                if not track[1] in true_running_car:
                                    true_running_car.append(track[1])
    return running_car, true_running_car
