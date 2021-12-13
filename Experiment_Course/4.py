import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Digital_Image_Processing_Code/lena.jpg')[:, :, ::-1]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


''' 一阶梯度算子
def def_robert(gray):
    # np.savetxt("result2.txt", gray, fmt="%d")
    h = gray.shape[0]
    w = gray.shape[1]
    #定义Robert算子的卷积核,这个是对角线方向
    x_kernal = np.array([[1,0],
                         [0,-1]])
    y_kernal = np.array([[0,1],
                         [-1,0]])
    #由于卷积核和图像进行卷积是以右下角的像素进行定位卷积核和目标像素点的位置关系，因此为了能够遍历整个图像，
    #需要在图像的第一行和第一列进行补零操作
    gray_zero = np.zeros((h+1,w+1))#先生成一个(h+1,w+1)的零图像矩阵
    #将原始图像去填充零矩阵，第一行和第一列不填充
    for i in range(1,h+1):
        for j in range(1,w+1):
            gray_zero[i,j]=gray[i-1,j-1]
    gray = gray_zero#将填充后的矩阵复制给gray
    #通过卷积，得到x和y两个方向的边缘检测图像
    x_edge = cv2.filter2D(gray, -1, x_kernal)
    y_edge = cv2.filter2D(gray, -1, y_kernal)
    #创建一个与原始图像大小一致的空图像，用于存放经过Robert算子的边缘图像矩阵
    edge_img = np.zeros((h,w),np.uint8)
    #根据计算公式得到最终的像素点灰度值
    for i in range(h):
        for j in range(w):
            edge_img[i,j] = (np.sqrt(x_edge[i,j]**2+y_edge[i,j]**2))
    return edge_img
robert_img = def_robert(img_gray)
plt.subplot(121), plt.axis('off'), plt.imshow(img_gray, cmap='gray')
plt.subplot(122), plt.axis('off'), plt.imshow(robert_img, cmap='gray')
plt.show()
'''

''' 二阶梯度算子
def def_prewitt(gray,type_flags):
    h = gray.shape[0]
    w = gray.shape[1]
    x_prewitt=np.array([[1,0,-1],
                       [1,0,-1],
                       [1,0,-1]])
    y_prewitt=np.array([[1,1,1],
                       [0,0,0],
                       [-1,-1,-1]])
 
    img=np.zeros([h+2,w+2])
    img[2:h+2,2:w+2]=gray[0:h,0:w]
    edge_x_img = cv2.filter2D(gray, -1, x_prewitt)
    edge_y_img = cv2.filter2D(gray, -1, y_prewitt)
 
    #p(i,j)=max[edge_x_img,edge_y_img]这里是将x,y中最大的梯度来代替该点的梯度
    edge_img_max=np.zeros([h,w],np.uint8)
    for i in range(h):
        for j in range(w):
            if edge_x_img[i][j]>edge_y_img[i][j]:
                edge_img_max=edge_x_img[i][j]
            else:
                edge_img_max=edge_y_img
 
    #p(i,j)=edge_x_img+edge_y_img#将梯度和替代该点梯度
    edge_img_sum=np.zeros([h,w],np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img_sum[i][j]=edge_x_img[i][j]+edge_y_img[i][j]
 
    # p(i,j)=|edge_x_img|+|edge_y_img|将绝对值的和作为梯度
    edge_img_abs = np.zeros([h, w],np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img_abs[i][j] = abs(edge_x_img[i][j]) + abs(edge_y_img[i][j])
 
 
    #p(i,j)=sqrt(edge_x_img**2+edge_y_img**2)将平方和根作为梯度
    edge_img_sqrt=np.zeros([h,w],np.uint8)
    for i in range(h):
        for j in range(w):
            edge_img_sqrt[i][j]=np.sqrt((edge_x_img[i][j])**2+(edge_y_img[i][j])**2)
 
 
    type = [edge_img_max,edge_img_sum,edge_img_abs,edge_img_sqrt]
    return type[type_flags]

prewitt_img = def_prewitt(img_gray,3)
plt.subplot(121), plt.axis('off'), plt.imshow(img_gray, cmap='gray')
plt.subplot(122), plt.axis('off'), plt.imshow(prewitt_img, cmap='gray')
plt.show()
'''

''' Laplacian of Gaussian算子
gray_lap = cv2.Laplacian(img_gray,cv2.CV_16S,ksize = 3)
dst = cv2.convertScaleAbs(gray_lap)
plt.subplot(121), plt.axis('off'), plt.imshow(img_gray, cmap='gray')
plt.subplot(122), plt.axis('off'), plt.imshow(dst, cmap='gray')
plt.show()
''' 
''' Canny 算子
edges = cv2.Canny(img_gray, 100, 200)
plt.subplot(121), plt.axis('off'), plt.imshow(img_gray, cmap='gray')
plt.subplot(122), plt.axis('off'), plt.imshow(edges, cmap='gray')
plt.show()
'''

''' OTSU法
ret1, th1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)  #方法选择为THRESH_OTSU
plt.subplot(121), plt.axis('off'), plt.imshow(img_gray, cmap='gray')
plt.subplot(122), plt.axis('off'), plt.imshow(th1, cmap='gray')
plt.show()
'''

''' 区域分割 分水岭算法
def watershed(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret0, thresh0 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh0,cv2.MORPH_OPEN,kernel, iterations = 2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret1, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # 查找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # 标记标签
    ret2, markers1 = cv2.connectedComponents(sure_fg)
    markers = markers1+1
    markers[unknown==255] = 0

    markers3 = cv2.watershed(img,markers)
    img[markers3 == -1] = [0,255,0]
    return thresh0,sure_bg,sure_fg,img

thresh0, sure_bg, sure_fg, img = watershed(img)
plt.subplot(221), plt.axis('off'), plt.imshow(img, cmap='gray')
plt.subplot(222), plt.axis('off'), plt.imshow(sure_bg, cmap='gray')
plt.subplot(223), plt.axis('off'), plt.imshow(sure_fg, cmap='gray')
plt.subplot(224), plt.axis('off'), plt.imshow(thresh0, cmap='gray')
plt.show()
'''
