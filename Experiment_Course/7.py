import os 
import cv2 
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 


datasets_path = './datasets/FaceDB_orl/'
label = []
data = []
for i in glob.glob(datasets_path+'*/*.png'):
    img = cv2.imread(i, 0)
    data.append(img.flatten())
    label.append(i.split('/')[-2])


C_data=np.array(data)
C_label=np.array(label)
print(C_data.shape, C_label.shape)


#print(C_data.shape)
#print(C_label)

#切分数据集
x_train,x_test,y_train,y_test=train_test_split(C_data,C_label,test_size=0.2,random_state=256)

pca=PCA(n_components=15,svd_solver='auto').fit(x_train)
#降维
x_train_pca=pca.transform(x_train)
x_test_pca=pca.transform(x_test)

svc=SVC(kernel='linear')
svc.fit(x_train_pca,y_train)
#测试识别准确度
print('%.5f'%svc.score(x_test_pca,y_test))