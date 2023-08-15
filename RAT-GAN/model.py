#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import math
from miscc.config import cfg
# In[ ]:


class NetG(nn.Module):
    def __init__(self, ngf=64, nz=100,lstm = None):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.lstm = lstm
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.fc = nn.Linear(nz, ngf*8*4*4)
        self.block0 = G_Block(ngf * 8, ngf * 8,lstm)#4x4
        self.block1 = G_Block(ngf * 8, ngf * 8,lstm)#4x4
        self.block2 = G_Block(ngf * 8, ngf * 8,lstm)#8x8
        self.block3 = G_Block(ngf * 8, ngf * 8,lstm)#16x16
        self.block4 = G_Block(ngf * 8, ngf * 4,lstm)#32x32
        self.block5 = G_Block(ngf * 4, ngf * 2,lstm)#64x64
        self.block6 = G_Block(ngf * 2, ngf * 1,lstm)#128x128

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )
        
    def forward(self, x, c):#x=(B，100) c=(B, 1024) 

        out = self.fc(x) #x=(B,ngf*8*4*4)

        out = out.view(x.size(0), 8*self.ngf, 4, 4)#x=(B,ngf*8,4,4)
        out = self.block0(out,c)#x=(B,ngf*8,4,4)

        out = F.interpolate(out, scale_factor=2)
        out = self.block1(out,c)#x=(B,ngf*8,8,8)

        out = F.interpolate(out, scale_factor=2)
        out = self.block2(out,c)#x=(B,ngf*8,16,16)

        out = F.interpolate(out, scale_factor=2)
        out = self.block3(out,c)#x=(B,ngf*8,32,32)

        out = F.interpolate(out, scale_factor=2)
        out = self.block4(out,c)#x=(B,ngf*4,64,64)

        out = F.interpolate(out, scale_factor=2)
        out = self.block5(out,c)#x=(B,ngf * 2,128,128)

        out = F.interpolate(out, scale_factor=2)
        out = self.block6(out,c)#x=(B,ngf * 1,256,256)

        out = self.conv_img(out)#x=(B,3,256,256)
        
        return out


# In[ ]:


class G_Block(nn.Module):

    def __init__(self, in_ch, out_ch,lstm):
        super(G_Block, self).__init__()
        self.lstm = lstm

        self.learnable_sc = in_ch != out_ch 
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)#输入通道，输出通道，卷积核大小，step，pad
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        self.affine1 = affine(in_ch)
        self.affine4 = affine(in_ch)
        self.affine2 = affine(out_ch)
        self.affine3 = affine(out_ch)
        self.affine5 = affine(out_ch)
        self.fea_l = nn.Linear(in_ch,256)
        self.fea_ll = nn.Linear(out_ch,256)
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def forward(self, x, y=None):#x=(B,n*8,4,4) y=(B,1024)
        return self.shortcut(x) + self.gamma * self.residual(x, y)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, yy=None):#x=(B,n*8,4,4) y=(B,1024)

        lstm_input = yy
        y,_  =  self.lstm(lstm_input)#y=(B,256)
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        

        lstm_input = yy
        y,_  =  self.lstm(lstm_input)        
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        
        
        
        h = self.c1(h)
        
 
        lstm_input = yy
        y,_  =  self.lstm(lstm_input)
        
        
        h = self.affine2(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)

        lstm_input = yy
        y,_  =  self.lstm(lstm_input)
        
        
        h = self.affine3(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
          
        return self.c2(h)


# In[ ]:


#(B,C,W,H) -> (B, nf, W, H)  256维的文本决定参数大小
class affine(nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(256, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):#x=(B,n*8,4,4) y=(B,256)
   
        weight = self.fc_gamma(y)#(B,n*8)
        bias = self.fc_beta(y) #(B,n*8)       

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)#增加一个维度  (l) -> (1,l)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias#(B,n*8,4,4)


# In[ ]:


class D_GET_LOGITS_att(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS_att, self).__init__()
        self.df_dim = ndf

        self.joint_conv = nn.Sequential(
            nn.Conv2d(ndf * 16+cfg.TEXT.EMBEDDING_DIM, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )
        self.block = resD(ndf * 16+cfg.TEXT.EMBEDDING_DIM, ndf * 16)#4

        self.joint_conv_att = nn.Sequential(
            nn.Conv2d(ndf * 16+cfg.TEXT.EMBEDDING_DIM, ndf * 2, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(ndf * 2, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )        
        self.softmax= nn.Softmax(2)
    def forward(self, out, y_):#(B,16ndf,8,8)   (B,1024)
     
        y = y_.view(-1, cfg.TEXT.EMBEDDING_DIM, 1, 1)# (B,1024,1,1)
        y = y.repeat(1, 1, 8, 8)#(B,1024,8,8)
        h_c_code = torch.cat((out, y), 1)#(B,1024+16ndf,8,8)
        p = self.joint_conv_att(h_c_code)#(B,1,8,8)       
        p = self.softmax(p.view(-1,1,64))#(B,1,64) 
        p = p.reshape(-1,1,8,8)#(B,1,8,8) 
        self.p = p
        p = p.repeat(1, cfg.TEXT.EMBEDDING_DIM, 1, 1)#(B,1024,8,8) 
        y = torch.mul(y,p)  #(B,1024,8,8) 
        h_c_code = torch.cat((out, y), 1)#(B,1024+16ndf,8,8) 
        h_c_code = self.block(h_c_code)#(B,16ndf,4,4) 

        y = y_.view(-1, cfg.TEXT.EMBEDDING_DIM, 1, 1)#(B,1024,1,1)
        y = y.repeat(1, 1, 4, 4)   #(B,1024,4,4)     
        h_c_code = torch.cat((h_c_code, y), 1)#(B,16ndf+1024,4,4) 
        out = self.joint_conv(h_c_code)#(B,1,4,4)
        return out


# In[ ]:


class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()

        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)#128
        self.block0 = resD(ndf * 1, ndf * 2)#64
        self.block1 = resD(ndf * 2, ndf * 4)#32
        self.block2 = resD(ndf * 4, ndf * 8)#16
        self.block3 = resD(ndf * 8, ndf * 16)#8
        self.block4 = resD(ndf * 16, ndf * 16)#4
        self.block5 = resD(ndf * 16, ndf * 16)#4

        self.COND_DNET = D_GET_LOGITS_att(ndf)

    def forward(self,x):#(B,3,256,256)

        out = self.conv_img(x)#(B,ndf,256,256)
        out = self.block0(out)#(B,2ndf,128,128)
        out = self.block1(out)#(B,4ndf,64,64)
        out = self.block2(out)#(B,8ndf,32,32)
        out = self.block3(out)#(B,16ndf,16,16)
        out = self.block4(out)#(B,16ndf,8,8)

        return out


# In[ ]:


class resD(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)


# In[ ]:


class CustomLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))#(1024,1024)
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))#(1024,1024)
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
        self.noise2h = nn.Linear(100,256)
        self.noise2c = nn.Linear(100,256)
        #self.init_hidden()
        self.hidden_seq = []
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def init_hidden(self,noise):
        h_t = self.noise2h(noise)
        c_t = self.noise2c(noise)

        self.c_t = c_t #(B,100,256)
        self.h_t = h_t #(B,100,256)
    def forward(self, x):#(2,1024)
        
        c_t = self.c_t 
        h_t = self.h_t
        HS = self.hidden_size
        x_t = x
        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + h_t @ self.U + self.bias#(B,1024)
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]), # [B,256]
            torch.sigmoid(gates[:, HS:HS*2]), # [B,256]
            torch.tanh(gates[:, HS*2:HS*3]),# [B,256]
            torch.sigmoid(gates[:, HS*3:]), # # [B,256]
        )
 
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        self.c_t = c_t
        self.h_t = h_t        

        return h_t, c_t

class Bert2Emb(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024,256)
    def forward(self, x):
        out = self.fc(x)
        return out