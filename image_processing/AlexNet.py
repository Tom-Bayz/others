import torch 
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96,kernel_size=11, stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256,kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_layers = nn.Sequential(
             nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
             nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
             nn.MaxPool2d(kernel_size=3, stride=2)
         )
        self.fc_layers = nn.Sequential(
             nn.Linear(in_features=9216, out_features=4096),
             nn.Linear(in_features=4096, out_features=4096),
             nn.Linear(in_features=4096, out_features=num_classes)
         )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, C*H*W]
        x = self.fc_layers(x)
        
        return self.softmax(x)
    

    
if __name__ == "__main__":
    # GPU
    device = torch.device("cuda")

    # DNN model
    net =  AlexNet(in_channels=1,num_classes=10).to(device)

    # make training data
    data = {}
    N = 1000
    data["image"] = torch.randn(N,1,227,227) # sample x channel x width x height
    data["label"] = torch.randn(N,10) # sample x #outputs
    data["index"] = np.array(range(N))

    # loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    # type of Gradient Decent
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optimizer

    # define #subsample 
    subsample_num = 100
    E = 1000

    loss_rec = pd.DataFrame(columns=["loss"],index=range(E))
    for epoch in tqdm(range(E)):

        random.shuffle(data["index"])
        running_loss = 0
        for i in (range(int(N/subsample_num))):
            subsample = data["index"][i*subsample_num:min((i+1)*subsample_num,len(data["index"])-1)]

            # forward prop
            x = data["image"][subsample,:,:,:].to(device)
            #print(x.shape)
            pred = net(x)
            #print(pred.shape)

            # target
            target = data["label"][subsample,:].to(device)

            # Gradient Decent
            optimizer.zero_grad()
            loss = criterion(pred,target)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        
        
        loss_rec.loc[epoch,"loss"] = running_loss

    
    loss_rec.plot()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.grid(True)
    plt.show()
    #

    

    