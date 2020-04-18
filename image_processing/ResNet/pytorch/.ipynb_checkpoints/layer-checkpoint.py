import torch 
import torch.nn as nn

# ResNetでよく使うConv関数を作成
def conv3x3(in_channels,out_channels,stride=1,groups=1,dilation=1):
    
    out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=True,
                      dilation=dilation)
    return out

def conv1x1(in_channels, out_channels, stride=1):
    
    out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
    
    return out


class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity_x = x  # hold input for shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity_x = self.downsample(x)
        
        out += identity_x  # shortcut connection
        
        return self.relu(out)
    

class ResidualLayer(nn.Module):

    def __init__(self, num_blocks, in_channels, out_channels, block=BasicBlock):
        super(ResidualLayer, self).__init__()
        downsample = None
        
        if in_channels != out_channels:
            downsample = nn.Sequential( conv1x1(in_channels, out_channels),
                                         nn.BatchNorm2d(out_channels)
                                      )
            
        self.first_block = block(in_channels, out_channels, downsample=downsample)
        self.blocks = nn.ModuleList(block(out_channels, out_channels) for _ in range(num_blocks))

    def forward(self, x):
        out = self.first_block(x)
        
        for block in self.blocks:
            out = block(out)
        
        return out