import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Digital_Image_Processing_Code/lena.jpg')[:, :, ::-1]
img2 = cv2.imread('./Digital_Image_Processing_Code/cat.jpg')[:, :, ::-1]

'''图像的读取操作
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plt.subplot(131), plt.title('Original Image'), plt.imshow(img)
plt.subplot(132), plt.title('Gray Image'), plt.imshow(gray, cmap='gray')
plt.subplot(133), plt.title('Binary Image'), plt.imshow(binary_img, cmap='gray')
'''

'''图像的基本运算
img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
add_image = cv2.add(img, img2)
sub_image = cv2.subtract(img, img2)
multiply_image = cv2.multiply(img, img2)
divid_image = cv2.divide(img, img2)

plt.subplot(231), plt.title('Original Image'), plt.imshow(img)
plt.subplot(232), plt.title('Original Image 2'), plt.imshow(img2)
plt.subplot(233), plt.title('Add Image'), plt.imshow(add_image)
plt.subplot(234), plt.title('Sub Image'), plt.imshow(sub_image)
plt.subplot(235), plt.title('Multiply Image'), plt.imshow(multiply_image)
plt.subplot(236), plt.title('Divid Image'), plt.imshow(divid_image)
plt.show()
'''

'''图像的几何变换
cols, rows, channels = img.shape
resize_img = cv2.resize(img, (cols // 2, rows // 2))
rotate_img = cv2.rotate(resize_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# 平移矩阵M：[[1,0,x],[0,1,y]]
M = np.float32([[1,0,100],[0,1,50]])
translation_img = cv2.warpAffine(img,M,(cols,rows))
crop_img = img[0:rows//2, 0:cols//2]
flip_img = cv2.flip(img, 1)

plt.subplot(231), plt.title('Original Image'), plt.imshow(img)
plt.subplot(232), plt.title('Resize Image'), plt.imshow(resize_img)
plt.subplot(233), plt.title('Rotate Image'), plt.imshow(rotate_img)
plt.subplot(234), plt.title('Translation Image'), plt.imshow(translation_img)
plt.subplot(235), plt.title('Crop Image'), plt.imshow(crop_img)
plt.subplot(236), plt.title('Flip Image'), plt.imshow(flip_img)
plt.show()
'''

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# y = k * x + b
bright_image = gray * 1.5 + 0
dark_image = gray * 0.75 + 0
negative_image = gray * -1 + 255
plt.subplot(221), plt.axis('off'), plt.title('Original Image'), plt.imshow(img)
plt.subplot(222), plt.axis('off'), plt.title('Bright Image'), plt.imshow(bright_image, cmap='gray', vmin=0, vmax=255)
plt.subplot(223), plt.axis('off'), plt.title('Dark Image'), plt.imshow(dark_image, cmap='gray', vmin=0, vmax=255)
plt.subplot(224), plt.axis('off'), plt.title('Negative Image'), plt.imshow(negative_image, cmap='gray', vmin=0, vmax=255)
plt.show()

