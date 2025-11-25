import torch
import torch.nn as nn
import torch.nn.functional as F

class DLICM(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=9, reduction_ratio=16):
        super(DLICM, self).__init__()
      

        # Depthwise: out_channels = in_channels (per-channel conv)

        self.depthwise = nn.Conv1d(in_channels,out_channels, kernel_size=3,dilation=dilation_rate,padding='same', groups=in_channels,bias=False)

        # pointwise: mixes channels, produces out_channels

        self.pointwise = nn.Conv1d(in_channels , out_channels , kernel_size=1 , bias=False)


        self.bn_dw = nn.BatchNorm1d(out_channels)
        self.bn_pw = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels)
 

        # SE block 

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),                          
            nn.Flatten(),                                     
            nn.Linear(out_channels,out_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear( out_channels // reduction_ratio, out_channels, bias=False),
            nn.Sigmoid() )

    def forward(self, x):
    
        # 1) depthwise (dilated)
        
        d = self.depthwise(x)
        d = self.bn_dw(d)
        d = self.relu(d)

        # 2) first residual: z = x + y  
        z = x + d

        # 3) pointwise on z

        p = self.pointwise(z)
        # p = self.bn_pw(p)
        # p = self.relu(p)

        # 4) SE

        se_weight = self.se(p).unsqueeze(-1) 
        p_weighted = p * se_weight                  

        # 5) final residual add with z
        out = p_weighted + p
        out = self.bn(out)
        out = self.relu(out)
        return out


class GAB(nn.Module):
    def __init__(self, in_channels):
        super(GAB, self).__init__()
       
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
 
        mask = self.softmax(self.conv(x))     
        gamma = (x * mask).sum(dim=1, keepdim=True)  

     

        return x + gamma


class DPCCNN(nn.Module):

    def __init__(self, num_classes):
        super(DPCCNN, self).__init__()
       
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(16, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.dlicm1 = DLICM(64, 64, dilation_rate=9, reduction_ratio=16)
        self.gab1 = GAB(64)
        self.dlicm2 = DLICM(64, 64, dilation_rate=9, reduction_ratio=16)
        self.gab2 = GAB(64)

        self.gap = nn.AdaptiveAvgPool1d(1)
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
        x = self.gab1(x)
        x = self.dlicm2(x)
        x = self.gab2(x)

        x = self.gap(x)            
        x = x.squeeze(-1)         
        x = self.fc(x)           
        return x