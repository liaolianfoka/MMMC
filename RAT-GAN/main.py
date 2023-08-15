#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import multiprocessing
from datasets import TextDataset
from model import NetG, NetD, CustomLSTM
from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file
import pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
multiprocessing.set_start_method('spawn', True)
UPDATE_INTERVAL = 200


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_e.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def sampling(test_dataloader, netG, netD, epoch, step_, device):
    split_dir = 'valid'
    # Build and load the generator
    # for coco wrap netG with DataParallel because it's trained on two 3090
    #    netG = nn.DataParallel(netG).cuda()
    #netG.load_state_dict(torch.load('../models/%s/netG_1543.pth' % (cfg.CONFIG_NAME)))

    netG.eval()
    netD.eval()
    diff_imgs = []
    diff_imgs_feature = []
    batch_size = cfg.TRAIN.BATCH_SIZE

    save_dir = '../%s/%s' % ("result", cfg.CONFIG_NAME)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for step, data in enumerate(test_dataloader, 0):
        imags, captions, class_ids, keys = data
        
        sent_emb = captions.detach()#取消梯度

        with torch.no_grad():
            noise = torch.randn(sent_emb.size()[0], 100)
            noise = noise.to(device)
            netG.lstm.init_hidden(noise)
            fake_imgs = netG(noise, sent_emb)
            fake_imgs_feature = netD(fake_imgs)
            diff_imgs.append(fake_imgs)
        for j in range(sent_emb.size()[0]):
            diff_imgs_feature.append(fake_imgs_feature[j].cpu())

    faker_imgs_feature_path = save_dir+"/faker_imgs_feature"
    faker_imgs_path = save_dir+"/faker_imgs"
    fake_imgs_png = torch.cat(diff_imgs, 0)
    if not os.path.exists(faker_imgs_feature_path):
        os.makedirs(faker_imgs_feature_path)
    if not os.path.exists(faker_imgs_path):
        os.makedirs(faker_imgs_path)

    vutils.save_image(fake_imgs_png.data,
                          faker_imgs_path+"/epoch_" + str(epoch) + "_" + str(step_) + ".png",
                          normalize=True)

    with open(faker_imgs_feature_path+ "/epoch_" + str(epoch) + "_" + str(step_) + ".pkl", "wb") as f:
        pickle.dump(diff_imgs_feature, f)


def train(train_dataloader, test_dataloader, netG, netD, optimizerG, optimizerD, state_epoch, batch_size, device):
    mkdir_p('../models/%s' % (cfg.CONFIG_NAME))
    if state_epoch > 0:
        print(f'Loading From Epoch {state_epoch} pth')
        netD.load_state_dict(torch.load(f'../models/{cfg.CONFIG_NAME}/netD_{state_epoch}.pth'))
        netG.load_state_dict(torch.load(f'../models/{cfg.CONFIG_NAME}/netG_{state_epoch}.pth'))
        optimizerD.load_state_dict(torch.load(f'../models/{cfg.CONFIG_NAME}/optimizerD_{state_epoch}.pth'))
        optimizerG.load_state_dict(torch.load(f'../models/{cfg.CONFIG_NAME}/optimizerG_{state_epoch}.pth'))

    for epoch in range(state_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1):
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            sampling(test_dataloader, netG, netD, epoch, 0, device)
            netG.train()
            netD.train()
        for step, data in enumerate(train_dataloader, 0):

            imags, captions, class_ids, keys = data
            sent_emb = captions.detach()#取消梯度
            imgs = imags[0].to(device)
            real_features = netD(imgs)#提取图像特征(B,16ndf,8,8)

            output = netD.COND_DNET(real_features, sent_emb)#融合图像和文本，(B,1,4,4)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()#损失 越大越好

            output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])#错位融合？
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()#损失 越小越好

            # synthesize fake images

            noise = torch.randn(batch_size, 100)
            noise = noise.to(device)
            netG.lstm.init_hidden(noise)
         
            fake = netG(noise, sent_emb)#假图像(B,3,256,256)

            # G does not need update with D
            fake_features = netD(fake.detach())#假图像特征(B,16ndf,8,8)

            errD_fake = netD.COND_DNET(fake_features, sent_emb)#融合假图像和文本(B,1,4,4)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()#正比
            if cfg.SINGLE:
                errD = errD_real + errD_fake
            else:
                errD = errD_real + (errD_fake + errD_mismatch) / 2.0
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            errD.backward()
            optimizerD.step()

            # MA-GP
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD(interpolated)
            out = netD.COND_DNET(features, sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, sent_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # update G
            features = netD(fake)
            output = netD.COND_DNET(features, sent_emb)
            errG = - output.mean()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                  % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(train_dataloader), errD.item(), errG.item()))
        
        #class_ids = [str(x) for x in class_ids.numpy().tolist()]
        """
        vutils.save_image(fake.data,
                          '%s/ep_%04d_%s.png' % ('../imgs', epoch, "_".join(class_ids)),
                          normalize=True)
        
        vutils.save_image(fake.data,
                          '%s/%s/ep_%04d.png' % ('../imgs', cfg.CONFIG_NAME, epoch),
                          normalize=True)
        """
        if epoch:
            torch.save(netG.state_dict(), f'../models/{cfg.CONFIG_NAME}/netG_{epoch}.pth')
            torch.save(netD.state_dict(), f'../models/{cfg.CONFIG_NAME}/netD_{epoch}.pth')
            torch.save(optimizerD.state_dict(), f'../models/{cfg.CONFIG_NAME}/optimizerD_{epoch}.pth')
            torch.save(optimizerG.state_dict(), f'../models/{cfg.CONFIG_NAME}/optimizerG_{epoch}.pth')
            try:
                os.remove(f'../models/{cfg.CONFIG_NAME}/netG_{epoch-1}.pth')
                os.remove(f'../models/{cfg.CONFIG_NAME}/netD_{epoch-1}.pth')
                os.remove(f'../models/{cfg.CONFIG_NAME}/optimizerD_{epoch-1}.pth')
                os.remove(f'../models/{cfg.CONFIG_NAME}/optimizerG_{epoch-1}.pth')
            except:
                print(f"Can't delete {epoch-1}.")


            with open(f"../models/{cfg.CONFIG_NAME}/latest.txt", mode='w') as progress:
                progress.write(str(f"{epoch}"))
            print(f"Saved checkpoint at Epoch: {epoch}.")
    return count


# In[ ]:


if __name__ == "__main__":
    args = parse_args()#读取运行指令的参数
    
    #假如--cfg不为空， 加载yml的设置
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    
    #假如gpu_id为-1，不使用gpu, 否则使用对应gpu
    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id
    
    #假如data_dir不为空， 使用对应数据集
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    
    #输出cfg
    print('Using config:')
    pprint.pprint(cfg)
    
    #假如为test模式, 设置随机数种子为100
    #假如为train模式且随机数种子为空， 则同样设为100
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100    
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    if not os.path.exists("../imgs/" + cfg.CONFIG_NAME):
        os.makedirs("../imgs/" + cfg.CONFIG_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.datetime.now(dateutil.tz.tzlocal())#获取时间
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')#设置时间格式
    output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)#设置模型参数保存路径
    torch.cuda.set_device(cfg.GPU_ID)#设置cuda的gpu
    cudnn.benchmark = True#为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    
    
    ######获取data loader####################################
    imsize = cfg.TREE.BASE_SIZE  #设置图片大小
    batch_size = cfg.TRAIN.BATCH_SIZE #设置batchz_size
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)), #短边统一(默认304)，另一边等比例缩放
        transforms.RandomCrop(imsize), #随机裁剪为imsize*imsize(默认256*256)
        transforms.RandomHorizontalFlip()])#依概率p水平翻转

    test_dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform,bert_path= cfg.BERT_PATH, device=device)     
    #assert test_dataset
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, drop_last=False,
            shuffle=False, num_workers=int(cfg.WORKERS))

    train_dataset = TextDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform, bert_path= cfg.BERT_PATH, device=device)
    #assert dataset
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    #############################################################
    
    lstm = CustomLSTM(cfg.TEXT.EMBEDDING_DIM, 256)
    netG = NetG(cfg.TRAIN.NF, 100, lstm).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)

    state_epoch = 0
    
    try:
        print(f"Read latest epoch from ../models/{cfg.CONFIG_NAME}/latest.txt")
        with open(f"../models/{cfg.CONFIG_NAME}/latest.txt", mode='r') as progress:
            state_epoch = int(progress.readline(8))
        print(f"Found latest epoch: {state_epoch}.")
    except:
        print("No latest.txt found.")
        pass
    
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))
    """
    if cfg.B_VALIDATION:
        count = sampling(netG, dataloader, device) 
        print('state_epoch:  %d' % (state_epoch))
    else:
    """
    count = train(train_dataloader,test_dataloader, netG, netD, optimizerG, optimizerD, state_epoch, batch_size, device)

