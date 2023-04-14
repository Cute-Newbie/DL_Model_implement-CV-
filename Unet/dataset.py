import os
import numpy as np
import torch
import torch.nn as nn


class Dataset(torch.utils.data.Dataset):

    def __init__(self,data_dir,transform = None): # 데이터 셋이 저장 되어 있는 경로와, transform

        self.data_dir = data_dir
        self.transform = transform 

        lst_data = os.listdir(self.data_dir) # dataset의 list를 얻을 수 있다. 

        lst_label = [f for f in lst_data if f.startswith("label")]
        lst_input = [f for f in lst_data if f.startswith("input")]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):

        return len(self.lst_label)
    
    def __getitem__(self,idx):

        label = np.load(os.path.join(self.data_dir,self.lst_label[idx]))
        input = np.load(os.path.join(self.data_dir,self.lst_input[idx]))

        label = label/255.0
        input = input/255.0

        #Neural Network에 들어가는 모든 input은 3개의 axis를 가져야 하는데 채널이 없는 경우에도 임의로 생성해야 한다.
        if label.ndim == 2:
            label = label[:,:,np.newaxis]

        if input.ndim == 2:
            input = input[:,:,np.newaxis]

        data = {'input':input,"label":label}

        if self.transform:
            data = self.transform(data)

        return data
    

##Transform 구현

class ToTensor():

    def __call__(self,data):
        label,input = data['label'],data['input']

        # Numpy의 array의 경우 Image의 차원이 (Y,X,CH)
        # Pytorch의 tensor의 경우 Image의 차원이 (CH,Y,X)
        
        

        label = label.transpose((2, 0, 1)).astype(np.float32)
        
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'input':torch.from_numpy(input),'label':torch.from_numpy(label)}

        return data
    
class Normalization():

    def __init__(self,mean = 0.5,std = 0.5):

        self.mean = mean
        self.std = std

    def __call__(self,data):

        label,input = data['label'],data['input']

        input = (input-self.mean)/self.std
        data = {"label":label,"input":input}
        return data
    

class RandomFlip():

    def __call__(self,data):

        label,input = data['label'],data['input']

        if np.random.rand()>0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand()>0.5:
            label = np.flipud(label)
            label = np.flipud(input)

        data = {"label":label,"input":input}

        return data 


    







    



