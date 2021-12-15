import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

img = cv2.imread('./Digital_Image_Processing_Code/lena.jpg')#[:, :, ::-1]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


origin_img = img.copy()
img1 = img.copy()
sift = cv2.SIFT_create()
kp = sift.detect(img_gray, None)
cv2.drawKeypoints(img_gray, kp, img)
cv2.drawKeypoints(img_gray, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(131), plt.axis('off'), plt.title("Original Image"), plt.imshow(origin_img[:, :, ::-1])
plt.subplot(132), plt.axis('off'), plt.title("SIFT"), plt.imshow(img, cmap='gray')
plt.subplot(133), plt.axis('off'), plt.title("SIFT"), plt.imshow(img1, cmap='gray')
plt.show()


MIN_MATCH_COUNT = 10 #设置最低特征点匹配数量为10
target = cv2.imread('./Digital_Image_Processing_Code/nose.png')#[:, :, ::-1]
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
kp1, des1 = sift.detectAndCompute(img_gray,None)
kp2, des2 = sift.detectAndCompute(target,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
#舍弃大于0.7的匹配
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #计算变换矩阵和MASK
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img_gray.shape
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    cv2.polylines(target,[np.int32(dst)],True,0,2, cv2.LINE_AA)
else:
    print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
draw_params = dict(matchColor=(0,255,0), 
                   singlePointColor=None,
                   matchesMask=matchesMask, 
                   flags=2)
result = cv2.drawMatches(img_gray,kp1,target,kp2,good,None,**draw_params)
plt.subplot(131), plt.axis('off'), plt.title("Original Image"), plt.imshow(origin_img[:, :, ::-1])
plt.subplot(132), plt.axis('off'), plt.title("Traget Image"), plt.imshow(target, cmap='gray')
plt.subplot(133), plt.axis('off'), plt.title("Result"), plt.imshow(result)
plt.show()
