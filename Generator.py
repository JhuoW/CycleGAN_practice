import torch
import torch.nn as nn
import torch.nn.functional as F
from resNet import ResNetBlock

class Generator(nn.Module):
    """
    输入形状 (batch_size, 3, 256,256)
    输出一张图片
    """
    def __init__(self):
        super(Generator,self).__init__()

        net = [nn.ReflectionPad2d(3),
               nn.Conv2d(3, 64, kernel_size=7),
               nn.InstanceNorm2d(64),
               nn.ReLU(inplace=True)]  # (batch_size, 64, 256, 256)

        # 下采样(增加通道减小shape)
        in_channel = 64
        net += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),  # output: (batch_size, 128, weith, height)

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding = 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]  

        # resBlock(9层):
        for _ in range(9):
            net += [
                ResNetBlock(256)
            ]
        
        # 上采样(减少通道增加shape)
        net += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(inplace = True),
            nn.ConvTranspose2d(128,64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]  # output (batch_size, 64, width, height)
        

        # output:
        net += [            
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
            ] # shape: (batch_size, 3, 256, 256)
        self.model = nn.Sequential(*net)

    def forward(self,x):
        return self.model(x)

        