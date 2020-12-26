from data import Apple2OrangeDataset, TRANSFORMS_train
from utils import tensor2image, LambdaLR, weights_init_normal, ReplayBuffer
import itertools
import tensorboardX
import glob
import torchvision.transforms as trForms
from torch.utils.data import Dataset, DataLoader, dataloader
import os
from PIL import Image
import random
from utils import *
from Generator import Generator
from Discriminator import Discriminator
import torch.nn as nn
import torch
import itertools
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载模型
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

netG_A2B.load_state_dict(torch.load("/data/zhuowei_common/models/cyclegan/netG_A2B.pth"))
netG_B2A.load_state_dict(torch.load("/data/zhuowei_common/models/cyclegan/netG_B2A.pth"))

netG_A2B.eval()
netG_B2A.eval()

size = 256

input_A = torch.ones([1,3,size, size], dtype=torch.float).to(device)

input_B = torch.ones([1,3,size, size], dtype=torch.float).to(device)

transforms_ = [
    trForms.ToTensor(),
    trForms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]


data_root = "/data/zhuowei/datasets/cyclegan/datasets/apple2orange"

dataloader = DataLoader(Apple2OrangeDataset(root = data_root,transform = transforms_, model = "test"), batch_size = 1, shuffle = False, num_workers = 8)

if not os.path.exists("/data/zhuowei_common/output/cyclegan/A"):
    os.mkdir("/data/zhuowei_common/output/cyclegan/A")
if not os.path.exists("/data/zhuowei_common/output/cyclegan/B"):
    os.mkdir("/data/zhuowei_common/output/cyclegan/B")
    
for i, batch in enumerate(dataloader):
    real_A = torch.tensor(input_A.copy_(batch["A"]), dtype = torch.float).to(device)
    real_B = torch.tensor(input_B.copy_(batch["A"]), dtype = torch.float).to(device)

    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)

    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    save_image(fake_A, "/data/zhuowei_common/output/cyclegan/A/{}.png".format(i))
    save_image(fake_B, "/data/zhuowei_common/output/cyclegan/B/{}.png".format(i))
    print(i)