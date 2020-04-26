import cv2
import numpy as np


DEBUG = False

#逆透视变换
def perspective(img):
    h, w= img.shape[:2]
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]])
    pts1 = np.float32([ [100,100],[0,h-1],[w-100,h-100],[w-1,0] ])
    M = cv2.getPerspectiveTransform(pts,pts1)
    dst = cv2.warpPerspective(img,M,(500,526))
    if DEBUG:
        cv2.imshow("dst", dst)
        cv2.waitKey(1000)
    return dst

#图像处理
def preprocessing(trans):
    gray = trans[:,:,0]
    gray = cv2.medianBlur(gray,5)
    ret, binary = cv2.threshold(gray, 143, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(binary,kernel,iterations = 1)
    kernel = np.ones((11,11),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    if DEBUG:
        cv2.imshow("binary", dilation)
        cv2.waitKey(1000)
    return dilation


def getGD(canny):
    sobelx=cv2.Sobel(canny,cv2.CV_32F,1,0,ksize=3)
    sobely=cv2.Sobel(canny,cv2.CV_32F,0,1,ksize=3)
    theta = np.arctan(np.abs(sobely/(sobelx+1e-10)))*180/np.pi
    Amplitude = np.sqrt(sobelx**2+sobely**2)
    mask = (Amplitude>30).astype(np.float32)
    Amplitude = Amplitude*mask
    return Amplitude, theta


def sliding_window(img1, img2, patch_size, istep=50):
    Ni, Nj = (int(s) for s in patch_size)
    for i in range(0, img1.shape[0] - Ni+1, istep):
        #for j in range(0, img1.shape[1] - Nj, jstep):
        #patch = (img1[i:i + Ni, j:j + Nj], img2[i:i + Ni, j:j + Nj])
        patch = (img1[i:i + Ni, 39:341], img2[i:i + Ni, 39:341])
        yield (i, 39), patch


def predict(patches, DEBUG):
    labels = np.zeros(len(patches))
    index = 0
    for Amplitude, theta in patches:
        mask = (Amplitude>25).astype(np.float32)
        h, b = np.histogram(theta[mask.astype(np.bool)], bins=range(0,80,5))
        low, high = b[h.argmax()], b[h.argmax()+1]
        newmask = ((Amplitude>25) * (theta<=high) * (theta>=low)).astype(np.float32)
        value = ((Amplitude*newmask)>0).sum()

        if value > 1000:
            labels[index] = 1
        index += 1
        if(DEBUG):
            print(h)
            print(low, high)
            print(value)
            cv2.imshow("newAmplitude", Amplitude*newmask)
            cv2.waitKey(1000)
    return labels


def getlocation(indices, labels, Ni, Nj):
    zc = indices[labels == 1]
    if len(zc) == 0:
        return 0, None
    else:
        xmin = int(min(zc[:,1]))
        ymin = int(min(zc[:,0]))
        xmax = int(xmin + Nj)
        ymax = int(max(zc[:,0]) + Ni)
        return 1, ((xmin, ymin), (xmax, ymax))


def zebra(img):
    Ni,Nj = (88,902)
    trans = perspective(img)
    gray = preprocessing(trans)
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)
    if DEBUG:
        cv2.imshow("canny", canny)
        cv2.waitKey(1000)
    Amplitude, theta = getGD(canny)
    if DEBUG:
        cv2.imshow("Amplitude", Amplitude)
        cv2.waitKey(1000)
    indices, patches = zip(*sliding_window(Amplitude, theta, patch_size=(Ni, Nj)))
    labels = predict(patches, DEBUG)
    indices = np.array(indices)
    ret, location = getlocation(indices, labels, Ni, Nj)
    #draw
    if DEBUG:
        for i, j in indices[labels == 1]:
            cv2.rectangle(img, (j, i), (j+Nj, i+Ni), (0, 0, 255), 3)
    if ret:
        cv2.rectangle(img, location[0], location[1], (255, 0, 255), 3)
        (xmin,ymin) = location[0]
        (xmax,ymax) = location[1]
    h = ymax - ymin
    a = 1/3 * h
    green =(0, 255, 0)
    cv2.line(img,(0, int(ymin + a)), (xmax, int(ymin + a)), green)
    return img
