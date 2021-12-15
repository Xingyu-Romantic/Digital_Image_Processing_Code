import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

img = cv2.imread('./Digital_Image_Processing_Code/figure.jpg')[:, :, ::-1]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



''' 开运算和闭运算
threshold = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
open_img = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
close_img = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
plt.subplot(131), plt.axis('off'), plt.title("Original Image"), plt.imshow(threshold, cmap='gray')
plt.subplot(132), plt.axis('off'), plt.title("Open Image"), plt.imshow(open_img, cmap='gray')
plt.subplot(133), plt.axis('off'), plt.title("Close Image"), plt.imshow(close_img, cmap='gray')
plt.show()
'''

''' 顶帽和黑帽算法
threshold = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))            
TOPHAT_img = cv2.morphologyEx(threshold, cv2.MORPH_TOPHAT, kernel)     
BLACKHAT_img = cv2.morphologyEx(threshold, cv2.MORPH_BLACKHAT, kernel) 
plt.subplot(131), plt.axis('off'), plt.title("Original Image"), plt.imshow(threshold, cmap='gray')
plt.subplot(132), plt.axis('off'), plt.title("TopHat Image"), plt.imshow(TOPHAT_img, cmap='gray')
plt.subplot(133), plt.axis('off'), plt.title("BlackHat Image"), plt.imshow(BLACKHAT_img, cmap='gray')
plt.show()
'''

'''  击中与不击中
src_A = cv2.imread('./Digital_Image_Processing_Code/A.png')
src_B = cv2.imread('./Digital_Image_Processing_Code/B.png')
src_BP = np.ones((src_B.shape[0]+10, src_B.shape[1]+10), dtype=np.uint8)  # 放大1
src_BP = src_BP*255
src_BP[5:src_B.shape[0]+5, 5:src_B.shape[1]+5] = 0  # 加外框
# src_BP = ~src_BP
src_BPC = ~src_BP
src_AC = ~src_A
erode_B_A = cv2.erode(src_A, src_BP)
erode_BPC_AC = cv2.erode(src_AC, src_BPC)
result = cv2.bitwise_and(erode_B_A, erode_BPC_AC)
plt.subplot(131), plt.axis('off'), plt.title("A"), plt.imshow(src_A, cmap='gray')
plt.subplot(132), plt.axis('off'), plt.title("B"), plt.imshow(src_B, cmap='gray')
plt.subplot(133), plt.axis('off'), plt.title("erode_BPC_AC"), plt.imshow(result, cmap='gray')
plt.show()
'''

''' 填充
img_gray = cv2.imread('./Digital_Image_Processing_Code/lena.jpg', 0)
retval, image = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
#image = ~image
image_C = cv2.bitwise_not(img_gray)
# 构造一个3×3的结构元素
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilate = cv2.dilate(img_gray, element)  #
erode = cv2.erode(img_gray, element)  # 腐蚀
# kernel形态运算的内核，一般由getStructuringElement函数生成，有三种形状
# MORPH_RECT 矩形
# MORPH_CROSS 交叉形
# MORPH_ELLIPSE 椭圆形

# 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
margin = cv2.absdiff(dilate, erode)
r, c = margin.shape[:2]

Mark_1 = np.zeros((r,c), np.uint8)  # 创建一个和图像一样的全零的图像矩阵
kernel = np.ones((7, 7), np.uint8)

epoch_img_1 = np.zeros((r,c), np.uint8)
epoch_img_2 = np.zeros((r,c), np.uint8)
Mark_1[200][250] = 255  # 种子点
stop = 0
while True:
    stop = stop+1
    print(stop)
    if stop == 50:
        epoch_img_1 = copy.deepcopy(Mark_1)
    if stop == 100:
        epoch_img_2 = copy.deepcopy(Mark_1)
    if stop == 200:
        # break
        pass
    Mark_tmep = copy.deepcopy(Mark_1)
    Mark_1 = cv2.dilate(Mark_1, kernel) #图像膨胀处理

    Mark_1 = cv2.bitwise_and(Mark_1, image)  # 图像交集
    flag = 0
    for i in range(r):
        for j in range(c):
            if Mark_1[i][j] == Mark_tmep[i][j]:
                flag = 0
            else:
                flag = 1
                break
        if flag == 1:
            break
    if flag == 0:
        break
plt.subplot(231), plt.axis('off'), plt.title("Original Image"), plt.imshow(img_gray, cmap='gray')
plt.subplot(232), plt.axis('off'), plt.title("Image_C"), plt.imshow(image_C, cmap='gray')
plt.subplot(233), plt.axis('off'), plt.title("Margin"), plt.imshow(margin, cmap='gray')
plt.subplot(234), plt.axis('off'), plt.title("epoch_img_1"), plt.imshow(epoch_img_1, cmap='gray')
plt.subplot(235), plt.axis('off'), plt.title("epoch_img_2"), plt.imshow(epoch_img_2, cmap='gray')
plt.subplot(236), plt.axis('off'), plt.title("Mark_1"), plt.imshow(Mark_1, cmap='gray')
plt.show()
'''