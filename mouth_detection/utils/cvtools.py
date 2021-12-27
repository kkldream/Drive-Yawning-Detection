import cv2
import math
import numpy as np

def imshow(win, img, rate):
    dsize = int(img.shape[1] * rate), int(img.shape[0] * rate)
    dimg = cv2.resize(img, dsize)
    cv2.imshow(win, dimg)

def sharpen(img, sigma=100):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    return usm

def convolution(img, kernel=-1):
    if kernel == -1:
        kernel = np.array([[-1, -1, 1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
    result = cv2.filter2D(img, -1, kernel)
    # result = cv2.filter2D(img, -1, kernel, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    return result

def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果

    brightness = 0
    contrast = -100 # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def rotate_img(img, angle, center_pos):
    h, w, _ = img.shape
    # center = (w // 2, h // 2) # 找到圖片中心
    center = center_pos
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))
    return rotate_img

def rotate_xy(pos, center_pos, angle):
	x = pos[0]
	y = pos[1]
	cx = center_pos[0]
	cy = center_pos[1]
	radian = angle * math.pi / -180
	x_new = int((x - cx) * math.cos(radian) - (y - cy) * math.sin(radian) + cx)
	y_new = int((x - cx) * math.sin(radian) + (y - cy) * math.cos(radian) + cy)
	return x_new, y_new

def get_distance(point_0, point_1):
    distance = math.pow((point_0[0] - point_1[0]), 2) + math.pow((point_0[1] - point_1[1]), 2)
    distance = math.sqrt(distance)
    return distance