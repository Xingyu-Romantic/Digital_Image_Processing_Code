import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Digital_Image_Processing_Code/lena.jpg')[:, :, ::-1]
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

''' 傅里叶变换
dft_img = cv2.dft(np.float32(img_gray),flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft_img)
reult = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))
center_shift = np.fft.fftshift(np.abs(dft_img))
dft_center_img = 20 * np.log(cv2.magnitude(center_shift[:, :, 0], center_shift[:, :, 1]))
plt.subplot(131), plt.axis('off'), plt.title('Original Image'), plt.imshow(img, cmap='gray')
plt.subplot(132), plt.axis('off'), plt.title('DFT Image'), plt.imshow(reult)
plt.subplot(133), plt.axis('off'), plt.title('DFT Center Image'), plt.imshow(dft_center_img)
plt.show()
'''

''' 离散余弦变换
dct_img = cv2.dct(np.float32(img_gray))
img_dct_log = np.log(abs(dct_img))
plt.subplot(121), plt.axis('off'), plt.title('Original Image'), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.axis('off'), plt.title('DCT Image'), plt.imshow(img_dct_log)
plt.show()
'''


''' Hough 变换
hough_image = img.copy()
edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100)
for line in lines:
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[0][2]
    y2 = line[0][3]
    cv2.line(hough_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
plt.subplot(121), plt.axis('off'), plt.title('Original Image'), plt.imshow(img)
plt.subplot(122), plt.axis('off'), plt.title('Hough Lines'), plt.imshow(hough_image)
plt.show()
'''