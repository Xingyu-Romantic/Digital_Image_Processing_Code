import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Digital_Image_Processing_Code/lena.jpg')[:, :, ::-1]

# 图像的平滑处理
''' 椒盐噪声
def addSaltNoise(img,snr):
    img_copy = img.copy()
    SNR = snr
    size = img.size
    noiseSize = int(size * (1 - SNR))
    for k in range(0, noiseSize):
        xi = int(np.random.uniform(0, img.shape[1]))
        xj = int(np.random.uniform(0, img.shape[0]))
        if random.random() < 0.5:
            img_copy[xj, xi] = 255
        else:
            img_copy[xj, xi] = 0
    return img_copy

Noise_img = addSaltNoise(img,0.8)
Medine_img = cv2.medianBlur(Noise_img,5) #中值滤波
Mean_img = cv2.blur(Noise_img,(5,5)) #均值滤波
plt.subplot(221), plt.axis('off'), plt.title('Original Image'), plt.imshow(img)
plt.subplot(222), plt.axis('off'), plt.title('Noise Image'), plt.imshow(Noise_img)
plt.subplot(223), plt.axis('off'), plt.title('Median Image'), plt.imshow(Medine_img)
plt.subplot(224), plt.axis('off'), plt.title('Mean Image'), plt.imshow(Mean_img)
plt.show()
'''

''' 高斯噪声
def gasuss_noise(img, mean=0, var=0.001):
    image = img.copy()
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

gasuss_img = gasuss_noise(img,0,0.02)
Median_img = cv2.medianBlur(gasuss_img,5) #中值滤波
Mean_img = cv2.blur(gasuss_img,(3,3)) #均值滤波
plt.subplot(221), plt.axis('off'), plt.title('Original Image'), plt.imshow(img)
plt.subplot(222), plt.axis('off'), plt.title('Gasuss Image'), plt.imshow(gasuss_img)
plt.subplot(223), plt.axis('off'), plt.title('Median Image'), plt.imshow(Median_img)
plt.subplot(224), plt.axis('off'), plt.title('Mean Image'), plt.imshow(Mean_img)
plt.show()
'''


''' 乘性噪声
Mulit_Nose_img = np.multiply(img, np.random.rand(img.shape[0], img.shape[1], img.shape[2]))
Guass_img = cv2.GaussianBlur(Mulit_Nose_img,(5,5), 0) / 255
plt.subplot(131), plt.axis('off'), plt.title('Original Image'), plt.imshow(img)
plt.subplot(132), plt.axis('off'), plt.title('Mulit Noise Image'), plt.imshow(Mulit_Nose_img)
plt.subplot(133), plt.axis('off'), plt.title('Guass Noise Image'), plt.imshow(Guass_img)
plt.show()
'''

# 图像的锐化处理
''' 直方图均衡化
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
Equal_Hist_img = cv2.equalizeHist(img)
Gray_Equal_Hist_img = cv2.calcHist([Equal_Hist_img], [0], None, [256], [0, 256])
plt.subplot(221), plt.axis('off'), plt.title('Original Image'), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(222), plt.axis('off'), plt.title('Histogram Image'), plt.plot(Hist_img)
plt.subplot(223), plt.axis('off'), plt.title('Equal Histogram Image'), plt.imshow(Equal_Hist_img, cmap='gray', vmin=0, vmax=255)
plt.subplot(224), plt.axis('off'), plt.title('Equal Histogram Image'), plt.plot(Gray_Equal_Hist_img)
plt.show()
'''

''' 平移后与平移前傅里叶频谱图对比
# 平移矩阵M：[[1,0,x],[0,1,y]]
cols, rows, channels = img.shape
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
M = np.float32([[1,0,100],[0,1,50]])
translation_img = cv2.warpAffine(img,M,(cols,rows))
dft_img = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft_img)
dft_img = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))
translation_dft_img = cv2.dft(np.float32(translation_img),flags=cv2.DFT_COMPLEX_OUTPUT)
translation_dftShift = np.fft.fftshift(translation_dft_img)
translation_dft_img = 20 * np.log(cv2.magnitude(translation_dftShift[:, :, 0], translation_dftShift[:, :, 1]))
plt.subplot(221), plt.axis('off'), plt.title('Original Image'), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(222), plt.axis('off'), plt.title('Translation Image'), plt.imshow(translation_img, cmap='gray', vmin=0, vmax=255)
plt.subplot(223), plt.axis('off'), plt.title('Dft Image'), plt.imshow(dft_img, cmap='gray', vmin=0, vmax=255)
plt.subplot(224), plt.axis('off'), plt.title('Translation Dft Image'), plt.imshow(translation_dft_img, cmap='gray', vmin=0, vmax=255)
plt.show()
'''

''' 旋转后与旋转前傅里叶频谱图对比
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
dft_img = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft_img)
dft_img = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))
rotated_dft_img = cv2.dft(np.float32(rotated_img),flags=cv2.DFT_COMPLEX_OUTPUT)
rotated_dftShift = np.fft.fftshift(rotated_dft_img)
rotated_dft_img = 20 * np.log(cv2.magnitude(rotated_dftShift[:, :, 0], rotated_dftShift[:, :, 1]))
plt.subplot(221), plt.axis('off'), plt.title('Original Image'), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(222), plt.axis('off'), plt.title('Rotated Image'), plt.imshow(rotated_img, cmap='gray', vmin=0, vmax=255)
plt.subplot(223), plt.axis('off'), plt.title('Dft Image'), plt.imshow(dft_img, cmap='gray', vmin=0, vmax=255)
plt.subplot(224), plt.axis('off'), plt.title('Rotated Dft Image'), plt.imshow(rotated_dft_img, cmap='gray', vmin=0, vmax=255)
plt.show()
'''

