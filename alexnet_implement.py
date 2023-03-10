# -*- coding: utf-8 -*-
"""AlexNet_implement.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1haTMb5mIeIS9sfrSn5arFyzXgGTfCXwb
"""

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
import matplotlib.pyplot as plt 
import torchvision.datasets as datasets
from datetime import datetime

"""2.Alexnet Implement

"""

# Model Implement 

class AlexNet(nn.Module):
    
    def __init__(self,n_classes):
        super().__init__()

    
        self.n_classes = n_classes

        self.extraction_layer = nn.Sequential(
            #input (227x227x3)

            #conv1
            nn.Conv2d(in_channels = 3, 
                      out_channels = 96,
                      kernel_size = 11,
                      stride =4,
                      padding = 0),
            
            nn.ReLU(),
            #maxpool1
            nn.MaxPool2d(kernel_size = 3,
                         stride = 2),
            
            #conv2
            nn.Conv2d(in_channels = 96,
                      out_channels = 256,
                      kernel_size = 5,
                      stride =1,
                      padding = 2),
            
            nn.ReLU(),
            #maxpool2
            nn.MaxPool2d(kernel_size = 3,
                         stride = 2),
            
            #conv3
            nn.Conv2d(in_channels = 256,
                      out_channels = 384,
                      kernel_size =3,
                      stride=1,
                      padding = 1),
            
            nn.ReLU(),

            #conv4
            nn.Conv2d(in_channels = 384,
                      out_channels = 384,
                      kernel_size =3,
                      stride=1,
                      padding = 1),
            nn.ReLU(),

            #conv5 
            nn.Conv2d(in_channels = 384,
                      out_channels = 256,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),

            nn.ReLU(),

            #Maxpool3

            nn.MaxPool2d(kernel_size = 3,
                         stride = 2),

            #drop_out 

            nn.Dropout(p = 0.5),
            
            
            )
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(6*6*256,2048),
            nn.ReLU(),
            nn.Linear(2048,1000),
            nn.ReLU(),
            nn.Linear(1000,n_classes)
            
        )

    def forward(self,x):
        extracted_image = self.extraction_layer(x)
        extracted_image = extracted_image.view(-1,-256*6*6)
        output = self.classifier_layer(extracted_image)
        probability = F.softmax(output,dim=1)


        return output,probability