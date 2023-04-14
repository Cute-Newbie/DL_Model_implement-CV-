##Packages Needed 

import os
import numpy as np
from PIL import Image ##PILLOW reads Tiff Files so that why we use PIL,Image
import matplotlib.pyplot as plt 


## Calling Data
dir_data = "/content/drive/MyDrive/Mnist_classifier/Unet/datasets"

name_label = "train-labels.tif"
name_input = "train-volume.tif"

#Imageopen() 함수의 리턴 값은 Image 객체 
img_label = Image.open(os.path.join(dir_data,name_label))
img_input = Image.open(os.path.join(dir_data,name_input))

ny,nx = img_label.size
nframe = img_label.n_frames


## Making Directories for train,val,test set
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data,"train")
dir_save_val = os.path.join(dir_data,"val")
dir_save_test = os.path.join(dir_data,"test")


if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)


## Shuffling Labels

id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## 
offset_nframe = 0

for i in range(nframe_train):
    # 이미지 형태를 읽어들일 때 frame단위로 읽어야 하는 경우 seek을 사용한다.
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    #np.save() -> 1개의 배열을 Numpy Format의 바이너리 파일로 저장 
    #np.load() -> np.save()로 저장 된 *.npy 파일을 배열로 불러오기 
    np.save(os.path.join(dir_save_train,"label_%03d.npy" % i),label_)
    np.save(os.path.join(dir_save_train,"input_%03d.npy" % i),input_)

##
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val,"label_%03d.npy" % i),label_)
    np.save(os.path.join(dir_save_val,"input_%03d.npy" % i),input_)


offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test,"label_%03d.npy" % i),label_)
    np.save(os.path.join(dir_save_test,"input_%03d.npy" % i),input_)











