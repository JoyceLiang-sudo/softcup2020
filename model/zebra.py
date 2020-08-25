import cv2


class Zebra:

    def __init__(self, up_zebra_line, down_zebra_line):
        """
        中线左坐标，中线右坐标，左上坐标，右下坐标
        """
        self.up_zebra_line = up_zebra_line
        self.down_zebra_line = down_zebra_line

    def update(self, possible_zebra):
        """
        更新斑马线
        """
        if len(possible_zebra) == 2:
            # 识别到了两根斑马线
            self.up_zebra_line = possible_zebra[0]
            self.down_zebra_line = possible_zebra[1]

        elif len(possible_zebra) == 1:
            # 只识别到了一根斑马线
            if self.up_zebra_line is None and self.down_zebra_line is not None:
                self.up_zebra_line = possible_zebra[0]
            elif self.up_zebra_line is None and self.down_zebra_line is None:
                self.down_zebra_line = possible_zebra[0]
            else:
                distance_1 = abs(self.up_zebra_line[0][1] - possible_zebra[0][0][1])
                distance_2 = abs(self.down_zebra_line[0][1] - possible_zebra[0][0][1])
                if distance_1 > distance_2:
                    self.down_zebra_line = possible_zebra[0]
                else:
                    self.up_zebra_line = possible_zebra[0]

    def draw_zebra_line(self, img, thickness=10):
        """
        画斑马线
        """
        if self.down_zebra_line is not None:
            # cv2.rectangle(img, self.down_zebra_line[2], self.down_zebra_line[3], (0, 255, 0), 3)
            cv2.line(img, self.down_zebra_line[0], self.down_zebra_line[1], (255, 200, 0), thickness)
        if self.up_zebra_line is not None:
            # cv2.rectangle(img, self.up_zebra_line[2], self.up_zebra_line[3], (0, 255, 0), 3)
            cv2.line(img, self.up_zebra_line[0], self.up_zebra_line[1], (255, 200, 0), thickness)

    def get_zebra_line(self, boxes, shape):
        """
        获得斑马线
        中线左坐标，中线右坐标，左上坐标，右下坐标
        """
        possible_zebra = []
        for box in boxes:
            if box[0] == 14:
                # 先扩大斑马线到左右边界
                possible_zebra.append([(0, box[2][1]), (shape[1], box[2][1]), (0, box[3][1]), (shape[1], box[4][1])])
        possible_zebra.sort()
        self.update(possible_zebra)
