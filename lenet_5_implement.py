# -*- coding: utf-8 -*-
"""LeNet-5_implement.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FySgVEKEW1Jsgrj2SNO0ctkRibwZD7Di
"""

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from torchvision import datasets,transforms
import matplotlib.pyplot as plt 

from datetime import datetime

"""1. Background knowledge of CNN that I have learned."""

# Torch Flatten 
t = torch.tensor([[[1, 2],
                  [3, 4]],
                 [[5, 6],
                  [7, 8]]])
print(t)
print(t.shape)
print()
t1 = torch.flatten(t)
print(t1)
print(t1.shape)
print()
t2 = torch.flatten(t, start_dim=1)
print(t2)
print(t2.shape)

"""2.LeNet5 Implement

"""

# Model Implement 

class LeNet5(nn.Module):
    
    def __init__(self,n_classes):
        super().__init__()

    
        self.n_classes = n_classes

        self.extraction_layer = nn.Sequential(
            nn.Conv2d(in_channels = 1, 
                      out_channels = 6,
                      kernel_size = 5,
                      stride =1),
            
            nn.Tanh(),
            
            nn.AvgPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 6,
                      out_channels = 16,
                      kernel_size = 5,
                      stride =1),
            
            nn.Tanh(),
            
            nn.AvgPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 16,
                      out_channels = 120,
                      kernel_size =5,
                      stride=1),
            
            nn.Tanh())
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(1*1*120,1*1*84),
            nn.Tanh(),
            nn.Linear(1*1*84,n_classes)
            
        )

    def forward(self,x):
        extracted_image = self.extraction_layer(x)
        extracted_image = extracted_image.view(extracted_image.size(0),-1)
        output = self.classifier_layer(extracted_image)
        probability = F.softmax(output,dim=1)


        return output,probability

"""3.Testing Model"""

# Transform image
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# Download Mnist dataset
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# Dataloader 
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

def train(train_loader,
          model,
          criterion,
          optimizer,
          epochs):
    
    model.train()
    train_loss = 0

    for epoch in range(epochs):

        for x_minibatch,y_minibatch in train_loader:

            optimizer.zero_grad()

            x_minibatch = x_minibatch.to(device)
            y_minibatch = y_minibatch.to(device)

            output,prob = model(x_minibatch)
            loss = criterion(output,y_minibatch)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        epoch_loss = train_loss / len(train_loader.dataset)
        print(f"{epoch+1}: {epoch_loss}")

    return model

model = LeNet5(n_classes = 10)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
device = "cuda" if torch.cuda.is_available() else 'cpu'
model.to(device)
criterion = nn.CrossEntropyLoss()
model_for_use = train(train_loader = train_loader,
          model = model,
          criterion = criterion,
          optimizer = optimizer,
          epochs = 15)

def validate(valid_loader,
             model,
             criterion):
    
    model.eval()
    correct = 0
    for x_minibatch,y_minibatch in valid_loader:

        x_minibatch = x_minibatch.to(device)
        y_minibatch = y_minibatch.to(device)

        output,prob = model(x_minibatch)
        labels = torch.argmax(prob,dim=1)
        correct += labels.eq(y_minibatch).sum().item()
    
    ratio = correct / len(valid_loader.dataset)

    #print(correct)
    print(f"accuracy : {ratio*100}%")

validate(valid_loader = valid_loader,
             model = model_for_use,
             criterion = criterion)