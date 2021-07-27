# 图像内插
import cv2
import numpy as np

def scale(img, x, y, rotate = 0):
    w, h, c = img.shape[:3]
    print(w, h)
    ret = np.zeros((x, y, c))
    x_scale = w // x
    y_scale = w // y
    for i in range(x):
        for j in range(y):
            ret[i, j,:] = img[i * x_scale, y * y_scale, :]
    return ret

img = cv2.imread('lena.jpg')
new_img = scale(img, 600, 600)
cv2.imshow('Original', img)
cv2.imshow('Scaled img', new_img / 255)
cv2.waitKey(0)
    


# 