import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from model import NetD, NetG

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=96)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--data_path', default='Textures/', help='folder to train data')
parser.add_argument('--outf', default='imgs/', help='folder to output images and model checkpoints')
opt = parser.parse_args()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize) # 创建一个空的标签向量
real_label = 1
fake_label = 0

D_Loss = [] # 鉴别器训练误差
G_Loss = [] # 生成器训练误差

def average_loss(list):
    '''
    用于求平均误差
    :param list: 每一次batch训练的误差
    :return: 平均误差值
    '''
    averageLoss = np.mean(list)
    return averageLoss

for epoch in range(1, opt.epoch + 1):
    D_err_temp = [] # 存放每一次batch的误差,
    G_err_temp = []
    for i, (imgs,_) in enumerate(dataloader):
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        ## 让D尽可能的把真图片判别为1
        imgs=imgs.to(device)
        output = netD(imgs)
        label.data.fill_(real_label) # 填充标签为1
        label=label.to(device)
        errD_real = criterion(output, label)
        errD_real.backward()
        ## 让D尽可能把假图片判别为0
        label.data.fill_(fake_label) # 填充标签为0
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise=noise.to(device)
        fake = netG(noise)  # 生成假图（64张）
        output = netD(fake.detach()) #避免梯度传到G，因为G不用更新，将fake单独从netG中抽离出来，共享数据，但是不具有梯度
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        errD.cpu()
        # 这些都是tensor类型的，必须转化为numpy类型来使用
        D_err_temp.append(errD.cpu().detach().numpy()) # 将数据单独分离出来用以作图：保存至cpu -> 分离 -> numpy化
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        errG = criterion(output, label)
        G_err_temp.append(errG.cpu().detach().numpy()) # 将数据单独分离出来用以作图
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    D_Loss.append(average_loss(D_err_temp))
    G_Loss.append(average_loss(G_err_temp))

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
    torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))

def show(D_error,G_error):
    '''
    画训练误差图，用来体现对抗
    :param D_error(list): 鉴别器结果误差
    :param G_error(list): 生成器结果误差
    :return:
    '''
    x1 = range(1,opt.epoch+1)
    x2 = range(1,opt.epoch+1)
    plt.plot(x1, D_error, label='D loss', linewidth=3, color='r', marker='o', markerfacecolor='blue', markersize=12)
    plt.plot(x2, G_error, label='G loss')
    plt.ylabel("Training Loss")
    plt.xlabel("Epoch")
    plt.title("Training Loss Visualization")
    plt.legend()
    plt.show()

show(D_Loss,G_Loss)