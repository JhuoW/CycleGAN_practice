import torch.nn as nn 
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """
    #先pad为了保证卷积前后大小不变
    步骤 pad->卷积->norm->relu -> pad
    """
    def __init__(self, in_channel):
        super(ResNetBlock,self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels= in_channel, out_channels= in_channel, kernel_size=3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),  # 直接修改上一层的值， 不常见新的变量
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channel,out_channels=in_channel, kernel_size = 3),
            nn.InstanceNorm2d(in_channel),

        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x) + x