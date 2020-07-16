import cv2
import numpy as np
from PIL import Image

DEBUG = False


class Zebra:

    def __init__(self, xmax, xmin, ymax, ymin):
        self.ymax = ymax
        self.xmax = xmax
        self.xmin = xmin
        self.ymin = ymin


def draw_zebra_line(img, zebra_line, thickness=2):
    """
    画斑马线
    """
    h = zebra_line.ymax - zebra_line.ymin
    a = 1 / 2 * h
    cv2.rectangle(img, (0, int(zebra_line.ymin)), (2 * int(zebra_line.xmax), int(zebra_line.ymax)), (0, 255, 0), 3)
    cv2.line(img, (0, int(zebra_line.ymin + a)), (2 * int(zebra_line.xmax), int(zebra_line.ymin + a)), (128, 0, 0),
             thickness)


def delete_contours(contours, delete_list):
    delta = 0
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours


def processing(img):
    # GRAY = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    GRAY = cv2.medianBlur(img, 5)
    RET, binary = cv2.threshold(GRAY, 115, 255, cv2.THRESH_BINARY)
    if DEBUG:
        cv2.imshow("binary", binary)
        cv2.waitKey(1000)
    kernel = np.ones((5, 5), np.uint8)
    erodtion = cv2.erode(binary, kernel, iterations=1)

    if DEBUG:
        cv2.imshow("erodtion", erodtion)
        cv2.waitKey(1000)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(erodtion, kernel, iterations=1)

    if DEBUG:
        cv2.imshow("dilation", dilation)
        cv2.waitKey(1000)
    canny = cv2.Canny(dilation, 50, 150, apertureSize=3)
    if DEBUG:
        cv2.imshow("canny", canny)
        cv2.waitKey(1000)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    threshold_low = 900

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < threshold_low:
            cv2.drawContours(canny, [contours[i]], 0, 0, -1)
    if DEBUG:
        cv2.imshow("img", canny)
        cv2.waitKey(1000)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        # 外接矩形框，没有方向角
        x, y, w, h = cv2.boundingRect(contours[i])
        if w / h >= 1.6:
            cv2.drawContours(canny, [contours[i]], 0, 0, -1)
    if DEBUG:
        cv2.imshow("img", canny)
        cv2.waitKey(1000)
    return canny


def get_GD(canny):
    sobelx = cv2.Sobel(canny, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(canny, cv2.CV_32F, 0, 1, ksize=3)
    theta = np.arctan(np.abs(sobely / (sobelx + 1e-10))) * 180 / np.pi
    Amplitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    mask = (Amplitude > 30).astype(np.float32)
    Amplitude = Amplitude * mask
    return Amplitude, theta


def sliding_window(img1, img2, patch_size, istep=30, jstep=30):
    Ni, Nj = (int(s) for s in patch_size)
    for i in range(0, img1.shape[0] - Ni + 1, istep):
        for j in range(0, img1.shape[1] - Nj, jstep):
            patch = (img1[i:i + Ni, j:j + Nj], img2[i:i + Ni, j:j + Nj])
            # patch = (img1[i:i + Ni, 39:341], img2[i:i + Ni, 39:341])
            yield (i, 39), patch


def predict(patches, DEBUG):
    labels = np.zeros(len(patches))
    index = 0
    for Amplitude, theta in patches:
        mask = (Amplitude > 25).astype(np.float32)
        h, b = np.histogram(theta[mask.astype(np.bool)], bins=range(0, 80, 5))
        low, high = b[h.argmax()], b[h.argmax() + 1]
        new_mask = ((Amplitude > 25) * (theta <= high) * (theta >= low)).astype(np.float32)
        value = ((Amplitude * new_mask) > 0).sum()

        if value > 1000:
            labels[index] = 1
        index += 1
        if (DEBUG):
            print(h)
            print(low, high)
            print(value)
            cv2.imshow("newAmplitude", Amplitude * new_mask)
            cv2.waitKey(1000)
    return labels


def get_location(indices, labels, Ni, Nj):
    zc = indices[labels == 1]
    if len(zc) == 0:
        return 0, None
    else:
        xmin = int(min(zc[:, 1]))
        ymin = int(min(zc[:, 0]))
        xmax = int(xmin + Nj)
        ymax = int(max(zc[:, 0]) + Ni)
        return 1, ((xmin, ymin), (xmax, ymax))


def get_zebra_line(img):
    global height, weight
    height = img.shape[0]
    weight = img.shape[1]
    Ni, Nj = (80, 902)
    low_hsv = np.array([0, 0, 0])
    high_hsv = np.array([125, 135, 120])
    mask = cv2.inRange(img, lowerb=low_hsv, upperb=high_hsv)
    if DEBUG:
        cv2.imshow("test", mask)
        cv2.waitKey(1000)
    gray = processing(mask)
    Amplitude, theta = get_GD(gray)
    if DEBUG:
        cv2.imshow("Amplitude", Amplitude)
        cv2.waitKey(1000)
    indices, patches = zip(*sliding_window(Amplitude, theta, patch_size=(Ni, Nj)))
    labels = predict(patches, DEBUG)
    indices = np.array(indices)
    ret, location = get_location(indices, labels, Ni, Nj)
    # if DEBUG:
    # for i, j in indices[labels == 1]:
    #    cv2.rectangle(img, (j, i), (j + Nj, i + Ni), (0, 0, 255), 3)
    if ret:
        # cv2.rectangle(img, location[0], location[1], (255, 0, 255), 3)
        (xmin, ymin) = location[0]
        (xmax, ymax) = location[1]
        return Zebra(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin)
    return Zebra(0, 0, 0, 0)
