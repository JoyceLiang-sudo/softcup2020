# coding=utf-8
def get_names(classes_path):
    """
    获得类别名称
    :param classes_path: 类别路径
    :return: 类别名称
    """
    import os
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_colors(class_names):
    """
    生成画矩形的颜色
    :param class_names: 类名称
    :return: 颜色
    """
    import colorsys
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    return colors
