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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = "/data/zhuowei/datasets/cyclegan/datasets/apple2orange"

batch_size = 1
size = 256
lr = 0.0002
n_epochs = 200
epoch = 0
decay_epoch = 100

train_loader = DataLoader(Apple2OrangeDataset(root, TRANSFORMS_train, model = "train"), batch_size=batch_size, shuffle=True, num_workers=10)

step = 0


##### network ######
netG_A2B = Generator().to(device)  # apple -> orange
netG_B2A = Generator().to(device)  # orange -> apple
netD_A = Discriminator().to(device) # 判断输入A的真假
netD_B = Discriminator().to(device) # 判断输入B的真假

loss_GAN = nn.MSELoss()
loss_cycle = nn.L1Loss()
loss_identity = nn.L1Loss()  # 真实数据和生成数据的相似度loss

opt_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr = lr, betas=(0.5,0.9999))

optD_A = torch.optim.Adam(netD_A.parameters(), lr = lr, betas = (0.5,0.9999))
optD_B = torch.optim.Adam(netD_B.parameters(), lr = lr, betas = (0.5, 0.9999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, 
                                                   lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optD_A, 
                                                   lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optD_B, 
                                                   lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

input_A = torch.ones([1,3,size,size], dtype= torch.float).to(device)
input_B = torch.ones([1,3,size,size], dtype = torch.float).to(device)
label_real = torch.ones([1], dtype = torch.float, requires_grad = False).to(device)
label_fake = torch.zeros([1], dtype = torch.float, requires_grad = False).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


log_path = "logs"
write_log = tensorboardX.SummaryWriter(log_path)  # 写入对应路径

for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch["A"]), dtype=torch.float).to(device)
        real_B = torch.tensor(input_B.copy_(batch["B"]), dtype=torch.float).to(device)

        opt_G.zero_grad()
        same_B = netG_A2B(real_B)
        loss_identity_B = loss_identity(same_B, real_B)  * 5.0

        same_A = netG_B2A(real_A)
        loss_identity_A = loss_identity(same_A, real_A)  * 5.0

        # G 要尽可能骗过 D
        fake_B = netG_A2B(real_A)
        pred_B = netD_B(fake_B)
        loss_GAN_A2B = loss_GAN(pred_B, label_real)

        fake_A = netG_B2A(real_B)
        pred_A = netD_A(fake_A)
        loss_GAN_B2A = loss_GAN(pred_A, label_real)

        # cycle loss
        recover_B = netG_A2B(fake_A)
        loss_cycle_A2B = loss_cycle(recover_B, real_B)

        recover_A = netG_B2A(fake_B)
        loss_cycle_B2A = loss_cycle(recover_A, real_A)

        loss_G = loss_identity_B + loss_identity_A + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A2B + loss_cycle_B2A

        loss_G.backward()

        opt_G.step()

        ######## Discriminator step #########
        optD_A.zero_grad()

        pred_real = netD_A(real_A)
        loss_D_real = loss_GAN(pred_real, label_real)

        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())  # 优化 loss_D_fake时不优化fake_A的参数 也就是不优化生成器
        loss_D_fake = loss_GAN(pred_fake, label_fake)
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5

        loss_D_A.backward()
        optD_A.step()


        optD_B.zero_grad()
        pred_real = netD_B(real_B)
        loss_D_real = loss_GAN(pred_real, label_real)
         
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = loss_GAN(pred_fake, label_fake)
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5

        loss_D_B.backward()
        optD_B.step()

        print("epoch: {}, loss G: {}, loss_G_identity: {}, loss_G_GAN: {}, loss_G_cycle: {}, loss_D_A: {}, loss_D_B: {} "\
                .format(epoch,
                        loss_G, 
                        loss_identity_A+loss_identity_B, 
                        loss_GAN_A2B+loss_GAN_B2A, 
                        loss_cycle_A2B + loss_cycle_B2A,
                        loss_D_A,
                        loss_D_B))
        
        write_log.add_scalar("loss_G", loss_G, global_step=step + 1)
        write_log.add_scalar("loss_G_identity", loss_identity_A+loss_identity_B, global_step=step + 1)
        write_log.add_scalar("loss_G_GAN",  loss_GAN_A2B+loss_GAN_B2A, global_step=step + 1)
        write_log.add_scalar("loss_G_cycle", loss_cycle_A2B + loss_cycle_B2A, global_step=step + 1)
        write_log.add_scalar("loss_D_A", loss_D_A, global_step=step + 1)
        write_log.add_scalar("loss_D_B", loss_D_B, global_step=step + 1)

        step += 1
    
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_G.step()

    torch.save(netG_A2B.state_dict(), "/data/zhuowei_common/models/cyclegan/netG_A2B.pth")
    torch.save(netG_B2A.state_dict(), "/data/zhuowei_common/models/cyclegan/netG_B2A.pth")
    torch.save(netD_A.state_dict(), "/data/zhuowei_common/models/cyclegan/netD_A.pth")
    torch.save(netD_B.state_dict(), "/data/zhuowei_common/models/cyclegan/netD_B.pth")

        




