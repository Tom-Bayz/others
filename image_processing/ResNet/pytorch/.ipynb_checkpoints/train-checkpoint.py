import torch
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from ResNet import *
import pickle

import pandas as pd

device = "cuda"
model = ResNet18(10).to(device)
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.001, 
                            momentum=0.9, 
                            weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor()])

data = {}
data["train"] = torchvision.datasets.MNIST('./data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

data["test"] = torchvision.datasets.MNIST('./data/',
                                             train=False,
                                             transform=transform,
                                             download=True)


train = DataLoader(data["train"],batch_size=1000,shuffle=True,drop_last=True)
test = DataLoader(data["test"],batch_size=1000)

result = pd.DataFrame(columns=["train loss","train accuracy","test loss","test accuracy"])


for epoch in range(10000):
    print("Epoch:",epoch)
    ####################################
    # training from here
    ####################################
    
    model.train()
    total = 0
    acc = 0
    result.loc[epoch,"train loss"] = 0
    for i, (image,label) in tqdm(enumerate(train)):

        # prepare input image and label
        image = image.to(device)
        label = label.to(device)

        # forward prop
        output = model(image)
        pred_label = torch.max(output,dim=1)[0]

        # Gradient Decent
        optimizer.zero_grad()
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        # record loss
        result.loc[epoch,"train loss"] += loss.item()

        total += label.size(0)
        acc += pred_label.eq(label).cpu().sum().item()

    result.loc[epoch,"train loss"] /= len(train)
    result.loc[epoch,"train accuracy"] = 100*acc/total
    
    
    ####################################
    # test from here
    ####################################    

    model.eval()
    total = 0
    acc = 0
    result.loc[epoch,"test loss"] = 0
    for i, (image,label) in tqdm(enumerate(test)):

        # prepare input image and label
        image = image.to(device)
        label = label.to(device)

        # forward prop
        output = model(image)
        pred_label = torch.max(output,dim=1)[0]

        # Gradient Decent
        optimizer.zero_grad()
        loss = criterion(output,label)

        # record loss
        result.loc[epoch,"test loss"] += loss.item()

        total += label.size(0)
        acc += pred_label.eq(label).cpu().sum().item()

    result.loc[epoch,"test loss"] /= len(test)
    result.loc[epoch,"test accuracy"] = 100*acc/total
    
    result.to_csv("ResNet_log.csv")
    
    with open(os.path.join("model","ResNet18_"+"%04d"%(epoch+1)+".mdl"),"wb") as f:
              pickle.dump(model,f)
    

