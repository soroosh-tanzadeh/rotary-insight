import torch
import torch.nn as nn
import torch.nn.functional as F

class DCIM(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=9):
        super(DCIM, self).__init__()
      

        # Depthwise: out_channels = in_channels (per-channel conv)

        self.depthwise = nn.Conv1d(in_channels,out_channels, kernel_size=3,dilation=dilation_rate,padding='same', groups=in_channels,bias=False)

        # pointwise: mixes channels, produces out_channels

        self.pointwise = nn.Conv1d(in_channels , out_channels , kernel_size=1 , bias=False)


        self.bn_dw = nn.BatchNorm1d(out_channels)
        self.bn_pw = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)
 
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
    
        # 1) depthwise (dilated)
        d = self.depthwise(x)
        d = self.bn_dw(d)
        d = self.relu(d)

      
        # 2) pointwise on z

        p = self.pointwise(d)
       
        out = self.bn(p)
        out = self.relu(out)
        # 3) golobal avrage pooling
        x = self.gap(out)            

        return x
    

class DSICNN(nn.Module):

    def __init__(self, num_classes):
        super(DSICNN, self).__init__()
       
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(16, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.dlicm1 = DCIM(64, 64, dilation_rate=9)

        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = x.unsqueeze(1)  

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.dlicm1(x)
        x = x.squeeze(-1)         
        x = self.fc(x)           
        return x    