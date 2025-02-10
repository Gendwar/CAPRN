import os
import sys
import cv2
import glob
import torch
import warnings
import math
import numpy as np
import shutil
from torch.optim import Adam
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import Data_load
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
import torchvision.transforms as Trans
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
class Downimage(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample1 = nn.AvgPool2d(kernel_size=2)
        self.downsample2 = nn.AvgPool2d(kernel_size=4)
        self.downsample3 = nn.AvgPool2d(kernel_size=8)
        self.downsample4 = nn.AvgPool2d(kernel_size=16)

    def forward(self, Img):

        return self.downsample4(Img), self.downsample3(Img), self.downsample2(Img), self.downsample1(Img), Img

class cross_attention(nn.Module):
    def __init__(self,inch,midch,downscale=2):
        super().__init__()
        self.down=nn.AvgPool2d(downscale)
        self.up = nn.Upsample(scale_factor=downscale, mode='bilinear', align_corners=True)
        self.inch=inch
        self.midch=midch
        # print(inch,midch)
        self.get_theta=nn.Conv2d(inch,midch,kernel_size=1,bias=False)
        self.get_phi  =nn.Conv2d(inch,midch,kernel_size=1,bias=False)
        self.get_g    =nn.Conv2d(inch,midch,kernel_size=1,bias=False)
        self.softmax  =nn.Softmax(dim=-1)
        self.outl=nn.Conv2d(midch,inch,kernel_size=3,padding=1,stride=1,bias=False)
    def forward(self,primary,cross):
        primary_down=self.down(primary)
        cross_down  =self.down(cross)
        b,c,w,h=primary_down.size()
        g=self.get_g(primary_down).view(b,-1,w*h)
        g=g.permute(0,2,1)      #bach*wh*16
        theta=self.get_theta(cross_down).view(b,-1,w*h)
        # print(cross.size(),g.size())
        phi=self.get_phi(primary_down).view(b,-1,w*h) #batch*16*wh
        theta=theta.permute(0,2,1)  #batch*wh*16
        theta_phi=torch.matmul(theta, phi)  #batch*wh*wh
        theta_phi=self.softmax(theta_phi)
        
        y=torch.matmul(theta_phi, g) #batch*wh*16
        y=y.permute(0,2,1).contiguous()# tensor变形后在内存中不连续，view会报错，contiguous深拷贝一份连续的变形后tensor
        y=y.view(b,-1,w,h)
        y=self.up(y)
        out=self.outl(y)
        return torch.cat((primary,out),dim=1)      
class double_conv(nn.Module):
    def __init__(self,inch,outch):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(inch, outch,  kernel_size=3,padding=1,stride=1,bias=False),
                                nn.InstanceNorm2d(outch),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(outch, outch, kernel_size=3,padding=1,stride=1,bias=False),
                                nn.InstanceNorm2d(outch),
                                nn.ReLU(inplace=True))
    def forward(self,x):
        out=self.conv(x)
        return out
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x
class Unet(nn.Module):
    def __init__(self,inch,convch=[16,32,64,128,256]):
        super().__init__()
        print('Creat Unet Encoder Model and convch: ',convch)
        # convch=[16,32,64,128,256]
        self.pool =nn.MaxPool2d(2)
        self.downimg=Downimage()
        self.en1=double_conv(inch, convch[0])
        self.en2=double_conv(convch[0], convch[1])
        self.en3=double_conv(convch[1], convch[2])
        self.en4=double_conv(convch[2], convch[3])
        self.en5=double_conv(convch[3], convch[4])
        self.up1=up_conv(convch[4], convch[3])
        self.de1=double_conv(convch[3]*2, convch[3])
        self.up2=up_conv(convch[3], convch[2])
        self.de2=double_conv(convch[2]*2, convch[2])
        self.up3=up_conv(convch[2], convch[1])
        self.de3=double_conv(convch[1]*2, convch[1])
        self.up4=up_conv(convch[1], convch[0])
        self.de4=double_conv(convch[0]*2, convch[0])
        self.out=nn.Conv2d(convch[0], out_channels=2, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
       
    def forward(self,fix,mov):
        fx0,fx1,fx2,fx3,fx4=self.downimg(fix)
        mx0,mx1,mx2,mx3,mx4=self.downimg(mov)
        bs=fix.size()[0]
        x=torch.cat((fix,mov),dim=0)
        e1=self.en1(x)
        e2=self.en2(self.pool(e1))
        e3=self.en3(self.pool(e2))
        e4=self.en4(self.pool(e3))
        e5=self.en5(self.pool(e4))  
        d0=e5
        d1=self.de1(torch.cat((e4,self.up1(d0)),dim=1))
        d2=self.de2(torch.cat((e3,self.up2(d1)),dim=1))
        d3=self.de3(torch.cat((e2,self.up3(d2)),dim=1))
        d4=self.de4(torch.cat((e1,self.up4(d3)),dim=1))
        # print(d1.size(),d2.size(),d3.size(),d4.size())
        fp0=torch.cat((d0[:bs,:,:,:],fx0),dim=1)
        mp0=torch.cat((d0[bs:,:,:,:],mx0),dim=1)
        fp1=torch.cat((d1[:bs,:,:,:],fx1),dim=1)
        mp1=torch.cat((d1[bs:,:,:,:],mx1),dim=1)
        fp2=torch.cat((d2[:bs,:,:,:],fx2),dim=1)
        mp2=torch.cat((d2[bs:,:,:,:],mx2),dim=1)
        fp3=torch.cat((d3[:bs,:,:,:],fx3),dim=1)
        mp3=torch.cat((d3[bs:,:,:,:],mx3),dim=1)
        fp4=torch.cat((d4[:bs,:,:,:],fx4),dim=1)
        mp4=torch.cat((d4[bs:,:,:,:],mx4),dim=1)
        return fp0,fp1,fp2,fp3,fp4,mp0,mp1,mp2,mp3,mp4  


class CAHRN(nn.Module):
    def __init__(self,inch=1,convch=[16,32,64,128,256],size=(512,512),modelflag='CAUnet'):
        super().__init__()
        self.enconder=Unet(inch=1,convch=convch)
        self.fca0=cross_attention(convch[4]+1,32,1)
        self.mca0=cross_attention(convch[4]+1,32,1)
        self.fca1=cross_attention(convch[3]+1,32,2)
        self.mca1=cross_attention(convch[3]+1,32,2)
        self.fca2=cross_attention(convch[2]+1,16,4)
        self.mca2=cross_attention(convch[2]+1,16,4)
        self.fca3=cross_attention(convch[1]+1,16,8)
        self.mca3=cross_attention(convch[1]+1,16,8)
        self.fca4=cross_attention(convch[0]+1,16,16)
        self.mca4=cross_attention(convch[0]+1,16,16)
        self.conv0=nn.Conv2d((convch[4]+1)*4,2,3,1,1,bias=False)
        self.conv1=nn.Conv2d((convch[3]+1)*4,2,3,1,1,bias=False)
        self.conv2=nn.Conv2d((convch[2]+1)*4,2,3,1,1,bias=False)
        self.conv3=nn.Conv2d((convch[1]+1)*4,2,3,1,1,bias=False)
        self.conv4=nn.Conv2d((convch[0]+1)*4,2,3,1,1,bias=False)
        self.stn1=SpatialTransformer([size[0]//8,size[1]//8])
        self.stn2=SpatialTransformer([size[0]//4,size[1]//4])
        self.stn3=SpatialTransformer([size[0]//2,size[1]//2])
        self.stn4=SpatialTransformer([size[0]   ,size[1]   ])
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        for m in self.modules():
        #     print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight,mean=0,std=1e-5)

    def forward(self,fix,mov):
        batchsize=fix.size()[0]
        fp0,fp1,fp2,fp3,fp4,mp0,mp1,mp2,mp3,mp4=self.enconder(fix,mov)
        
        self.warpl=[]
        fp0_ca=self.fca0(fp0,mp0)
        mp0_ca=self.mca0(mp0,fp0)
        w0=self.conv0(torch.cat((fp0_ca,mp0_ca),dim=1))
        self.warpl.append(w0)
        flow=self.up(2*w0)
        
        mp1=self.stn1(mp1,flow)
        fp1_ca=self.fca1(fp1,mp1)
        mp1_ca=self.mca1(mp1,fp1)
        w1=self.conv1(torch.cat((fp1_ca,mp1_ca),dim=1))
        self.warpl.append(w1)
        flow=self.up(2*(self.stn1(flow,w1)+w1))
       
        mp2=self.stn2(mp2,flow)
        fp2_ca=self.fca2(fp2,mp2)
        mp2_ca=self.mca2(mp2,fp2)
        w2=self.conv2(torch.cat((fp2_ca,mp2_ca),dim=1))
        self.warpl.append(w2)
        flow=self.up(2*(self.stn2(flow,w2)+w2))
        
        mp3=self.stn3(mp3,flow)
        fp3_ca=self.fca3(fp3,mp3)
        mp3_ca=self.mca3(mp3,fp3)
        w3=self.conv3(torch.cat((fp3_ca,mp3_ca),dim=1))
        self.warpl.append(w3)
        flow=self.up(2*(self.stn3(flow,w3)+w3))
        
        mp4=self.stn4(mp4,flow)
        fp4_ca=self.fca4(fp4,mp4)
        mp4_ca=self.mca4(mp4,fp4)
        w4=self.conv4(torch.cat((fp4_ca,mp4_ca),dim=1))
        self.warpl.append(w4)
        flow=self.stn4(flow,w4)+w4
        # warped=self.stn4(mov,flow)
        return flow,self.warpl
class CAPR(nn.Module):
    def __init__(self,imgsize=[512,512]):
        super().__init__()
        self.net=CAHRN()
        self.STN=SpatialTransformer(imgsize)
    def forward(self,fix,mov):
        flow,warpl=self.net(fix,mov)
        warped=self.STN(mov,flow)
        return warped,warpl
        # return warped,flow
    