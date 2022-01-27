# -*- coding: utf-8 -*-

import torch.nn as nn
#import torch.nn.functional as F
import torch
#import numpy as np
#from torch.autograd import Variable

from torchvision.models import resnet50

class Resnet50(nn.Module):
    def __init__(self, mid_dim, last_dim, dropout):
        super(Resnet50, self).__init__()
        resnet = resnet50(pretrained=True)

        # ct = 0
        # for child in resnet.children():
        #     ct += 1
        #     if ct < 8:
        #         for param in child.parameters():
        #             param.requires_grad = False
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, mid_dim), 
            nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(mid_dim, last_dim)

        )
    
    def forward(self, x):
        # with torch.no_grad():
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

class PixelTransformerResnet(nn.Module):#(pixel transformer in resnet)
    def __init__(self, mid_dim = 256, last_dim =3, dropout = 0.1, n_heads = 8):
        super(PixelTransformerResnet, self).__init__()
        resnet = resnet50(pretrained=True)
        
        ct = 0
        
        for child in resnet.children():
            ct += 1
            if ct < 2:
                for param in child.parameters():
                    param.requires_grad = False
        
        self.feature_extractor_before_pixelTransformer = nn.Sequential(*list(resnet.children())[:-3]) 
        self.feature_extractor_after_pixelTransformer = nn.Sequential(*list(resnet.children())[-2:-1])
        
        # self.pixelTransfomer6_1 = PixelTransformer(512, 64, 512, n_heads)
        # self.pixelTransfomer6_2 = PixelTransformer(512, 64, 512, n_heads)
        # self.pixelTransfomer6_3 = PixelTransformer(512, 64, 512, n_heads)
        # self.pixelTransfomer6_4 = PixelTransformer(512, 64, 512, n_heads)
        # self.pixelTransfomer6_5 = PixelTransformer(512, 64, 512, n_heads)
        # self.pixelTransfomer6_6 = PixelTransformer(512, 64, 512, n_heads)
        
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(512, 1024, kernel_size=1, stride=2, padding=0), 
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU()
        #     #, nn.LogSoftmax(dim = 1)
        #   )     
        
        self.pixelTransfomer7_1 = PixelTransformer(1024, 128, 1024, n_heads)
        self.pixelTransfomer7_2 = PixelTransformer(1024, 128, 1024, n_heads)
        self.pixelTransfomer7_3 = PixelTransformer(1024, 128, 1024, n_heads)
        self.final = nn.Sequential(
            nn.Linear(1024, mid_dim), 
            nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(mid_dim, last_dim)
            #, nn.LogSoftmax(dim = 1)
            )
    def forward(self, x):
        
        #with torch.no_grad():
        x = self.feature_extractor_before_pixelTransformer(x)#size = batch*1024*16*16
        #input_x = x
        
        # x = self.pixelTransfomer6_1(x)# size = batch* 512 *16*16
        # x = self.pixelTransfomer6_2(x)# size = batch* 512 *16*16
        # x = self.pixelTransfomer6_3(x)# size = batch* 512 *16*16
        # x = self.pixelTransfomer6_4(x)# size = batch* 512 *16*16
        # x = self.pixelTransfomer6_5(x)# size = batch* 512 *16*16
        # x = self.pixelTransfomer6_6(x)# size = batch* 512 *16*16
        
        # x = self.conv6(x)
        
        x = self.pixelTransfomer7_1(x)# size = batch* 512 *16*16
        x = self.pixelTransfomer7_2(x)# size = batch* 512 *16*16
        x = self.pixelTransfomer7_3(x)# size = batch* 512 *16*16
        #x = x + input_x
        x = self.feature_extractor_after_pixelTransformer(x)# size = batch* 512 *1*1
        x = x.view(x.size(0), -1)#size = batch *512
        x = self.final(x)
        return x
        
        
class PixelTransformer(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, n_heads):
        super(PixelTransformer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels*n_heads, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, mid_channels*n_heads, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, mid_channels*n_heads, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels*n_heads)
        self.bn2 = nn.BatchNorm2d(mid_channels*n_heads)
        self.bn3 = nn.BatchNorm2d(mid_channels*n_heads)
        self.bn4 = nn.BatchNorm2d(mid_channels*n_heads)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.dm = DislocationMultiplication()
        self.conv4 = nn.Conv2d(mid_channels*n_heads, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        
        identity = x
        q_q = self.conv1(x)#size = batch*(nheads*512)*16*16        
        k_k = self.conv2(x)
        v_v = self.conv3(x)
        
        q_q = self.bn1(q_q)
        k_k = self.bn2(k_k)
        v_v = self.bn3(v_v)
        q_q = self.relu(q_q)
        k_k = self.relu(k_k)
        v_v = self.relu(v_v)
        
        out = self.dm(q_q, k_k, v_v)
        out = self.bn4(out)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.bn5(out)

        out += identity
        out = self.relu(out)
        #out = self.dm
        #print(q.shape)
        #print(k.shape)
        #print(v.shape)
        #print(q_q.shape)
        #print(k_k.shape)
        #print(v_v.shape)
        return out
        
class DislocationMultiplication(nn.Module):
    def __init__(self, ):
        super(DislocationMultiplication, self).__init__()
        #self.pad = nn.ZeroPad2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.pad = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        
        
        self.padvLeftTop = nn.ZeroPad2d(padding=(1, 0, 1, 0))
        self.padvTop = nn.ZeroPad2d(padding=(0, 0, 1, 0))
        self.padvRightTop = nn.ZeroPad2d(padding=(0, 1, 1, 0))
        self.padvLeft = nn.ZeroPad2d(padding=(1, 0, 0, 0))
        self.padvRight = nn.ZeroPad2d(padding=(0, 1, 0, 0))
        self.padvLeftDown = nn.ZeroPad2d(padding=(1, 0, 0, 1))
        self.padvDown = nn.ZeroPad2d(padding=(0, 0, 0, 1))
        self.padvRightDown = nn.ZeroPad2d(padding=(0, 1, 0, 1))
    def forward(self, q, k, v):
        qkLeftTop = self.kLeftTop(q, k)
        qkRightTop = self.kRightTop(q, k)# size= batch*16*16
        qkLeftDown = self.kLeftDown(q, k)
        qkRightDown = self.kRightDown(q, k)
        qkLeft = self.kLeft(q, k)
        qkTop = self.kTop(q, k)
        qkRight = self.kRight(q, k)
        qkDown = self.kDown(q, k)
        qkCenter = self.kCenter(q, k)
        
        B, C, H, W = q.shape

                
        qkLeftTop = qkLeftTop.view(qkLeftTop.size(0), -1).unsqueeze(2)
        qkRightTop = qkRightTop.view(qkRightTop.size(0), -1).unsqueeze(2)
        qkLeftDown = qkLeftDown.view(qkLeftDown.size(0), -1).unsqueeze(2)
        qkRightDown = qkRightDown.view(qkRightDown.size(0), -1).unsqueeze(2)
        qkLeft = qkLeft.view(qkLeft.size(0), -1).unsqueeze(2)
        qkTop = qkTop.view(qkTop.size(0), -1).unsqueeze(2)
        qkRight = qkRight.view(qkRight.size(0), -1).unsqueeze(2)
        qkDown = qkDown.view(qkDown.size(0), -1).unsqueeze(2)
        qkCenter = qkCenter.view(qkCenter.size(0), -1).unsqueeze(2)
        
        qkCat = torch.cat([qkLeftTop, qkTop, qkRightTop, qkLeft, qkCenter, qkRight, qkLeftDown, qkDown, qkRightDown], 2)
        att = qkCat/(C**0.5) 
        att = self.softmax(att)
 
        
        vLeftTop = self.padvLeftTop(v)[:,:,:-1,:-1] 
        vTop = self.padvTop(v)[:,:,:-1,:]
        vRightTop = self.padvRightTop(v)[:,:,:-1,1:]
        vLeft = self.padvLeft(v)[:,:,:,:-1]
        vCenter = v
        vRight = self.padvRight(v)[:,:,:,1:]
        vLeftDown = self.padvLeftDown(v)[:,:,1:,:-1]
        vDown = self.padvDown(v)[:,:,1:,:]
        vRightDown = self.padvRightDown(v)[:,:,1:,1:]
        
        
        vLeftTop = vLeftTop.contiguous().view(vLeftTop.size(0), vLeftTop.size(1), -1).unsqueeze(3)
        vTop = vTop.contiguous().view(vTop.size(0), vTop.size(1), -1).unsqueeze(3)
        vRightTop = vRightTop.contiguous().view(vRightTop.size(0), vRightTop.size(1), -1).unsqueeze(3)
        vLeft = vLeft.contiguous().view(vLeft.size(0), vLeft.size(1), -1).unsqueeze(3)
        vCenter = vCenter.contiguous().view(vCenter.size(0), vCenter.size(1), -1).unsqueeze(3)
        vRight = vRight.contiguous().view(vRight.size(0), vRight.size(1), -1).unsqueeze(3)
        vLeftDown = vLeftDown.contiguous().view(vLeftDown.size(0), vLeftDown.size(1), -1).unsqueeze(3)
        vDown = vDown.contiguous().view(vDown.size(0), vDown.size(1), -1).unsqueeze(3)
        vRightDown = vRightDown.contiguous().view(vRightDown.size(0), vRightDown.size(1), -1).unsqueeze(3)
        vCat = torch.cat([vLeftTop, vTop, vRightTop, vLeft, vCenter, vRight, vLeftDown, vDown, vRightDown], 3) 
        #print(vCat.shape)
        att = att.unsqueeze(1).repeat(1, vCat.size(1), 1, 1)
        #print(att)
        v = torch.mul(vCat, att)
       # print(v.shape)
        v = torch.sum(v, dim = 3)   #B*C*256
        #print(v.shape)
        v = v.view(v.size(0), v.size(1), H, W)
        
        
        return v
                
                
                                
                
                

        
    def kLeftTop(self, q, k):
        
        B, C, H, W = q.shape

        paddingRowQ = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnQ = torch.zeros((B, C, H+1, 1)).cuda()
        paddingRowK = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnK = torch.zeros((B, C, H+1, 1)).cuda()
    
        afterPaddingQ1 = torch.cat((q, paddingRowQ),2)
        afterPaddingQ2 = torch.cat((afterPaddingQ1, paddingColumnQ),3)
        
        afterPaddingK1 = torch.cat((paddingRowK, k),2)
        afterPaddingK2 = torch.cat((paddingColumnK, afterPaddingK1),3)      
        
        out = afterPaddingQ2*afterPaddingK2
        out = out[:,:,0:H, 0:W]
        out = torch.sum(out, dim = 1) 
        return out
        
    def kRightTop(self, q, k):

        B, C, H, W = q.shape

        paddingRowQ = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnQ = torch.zeros((B, C, H+1, 1)).cuda()
        paddingRowK = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnK = torch.zeros((B, C, H+1, 1)).cuda()
    
        afterPaddingQ1 = torch.cat((q, paddingRowQ),2)
        afterPaddingQ2 = torch.cat((paddingColumnQ, afterPaddingQ1),3)
        
        afterPaddingK1 = torch.cat((paddingRowK, k),2)
        afterPaddingK2 = torch.cat((afterPaddingK1, paddingColumnK),3)      
        
        out = afterPaddingQ2*afterPaddingK2
        out = out[:,:,0:H, 1:W+1]
        out = torch.sum(out, dim = 1) 
        return out
    def kLeftDown(self, q, k):

        B, C, H, W = q.shape

        paddingRowQ = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnQ = torch.zeros((B, C, H+1, 1)).cuda()
        paddingRowK = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnK = torch.zeros((B, C, H+1, 1)).cuda()
    
        afterPaddingQ1 = torch.cat((paddingRowQ, q),2)
        afterPaddingQ2 = torch.cat((afterPaddingQ1, paddingColumnQ),3)
        
        afterPaddingK1 = torch.cat((k, paddingRowK),2)
        afterPaddingK2 = torch.cat((paddingColumnK, afterPaddingK1),3)      
        
        out = afterPaddingQ2*afterPaddingK2
        out = out[:, :, 1:H+1, 0:W]
        out = torch.sum(out, dim = 1) 
        return out
    def kRightDown(self, q, k):

        B, C, H, W = q.shape

        paddingRowQ = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnQ = torch.zeros((B, C, H+1, 1)).cuda()
        paddingRowK = torch.zeros((B, C, 1, W)).cuda()
        paddingColumnK = torch.zeros((B, C, H+1, 1)).cuda()
    
        afterPaddingQ1 = torch.cat((paddingRowQ, q),2)
        afterPaddingQ2 = torch.cat((paddingColumnQ, afterPaddingQ1),3)
        
        afterPaddingK1 = torch.cat((k, paddingRowK),2)
        afterPaddingK2 = torch.cat((afterPaddingK1, paddingColumnK),3)      
        
        out = afterPaddingQ2*afterPaddingK2
        out = out[:,:,1:H+1, 1:W+1]
        out = torch.sum(out, dim = 1) 
        return out
    def kLeft(self, q, k):

        B, C, H, W = q.shape

        #paddingRowQ = torch.zeros((B, C, 1, W))
        paddingColumnQ = torch.zeros((B, C, H, 1)).cuda()
        #paddingRowK = torch.zeros((B, C, 1, W))
        paddingColumnK = torch.zeros((B, C, H, 1)).cuda()
    
        #afterPaddingQ1 = torch.cat((q, paddingRowQ),2)
        afterPaddingQ = torch.cat((q, paddingColumnQ),3).cuda()
        
        #afterPaddingK1 = torch.cat((paddingRowK, k),2)
        afterPaddingK = torch.cat((paddingColumnK, k),3).cuda()    
        
        out = afterPaddingQ*afterPaddingK
        out = out[:, :, :, 0:W]
        out = torch.sum(out, dim = 1) 
        return out
    
    def kTop(self, q, k):

        B, C, H, W = q.shape

        paddingRowQ = torch.zeros((B, C, 1, W)).cuda()
        #paddingColumnQ = torch.zeros((B, C, H+1, 1))
        paddingRowK = torch.zeros((B, C, 1, W)).cuda()
        #paddingColumnK = torch.zeros((B, C, H+1, 1))
    
        afterPaddingQ = torch.cat((q, paddingRowQ),2).cuda()
        #afterPaddingQ2 = torch.cat((afterPaddingQ1, paddingColumnQ),3)
        
        afterPaddingK = torch.cat((paddingRowK, k),2).cuda()
        #afterPaddingK2 = torch.cat((paddingColumnK, afterPaddingK1),3)      
        
        out = afterPaddingQ*afterPaddingK
        out = out[:, :, 0:H, :]
        out = torch.sum(out, dim = 1) 
        return out
    
    def kRight(self, q, k):

        B, C, H, W = q.shape

        #paddingRowQ = torch.zeros((B, C, 1, W))
        paddingColumnQ = torch.zeros((B, C, H, 1)).cuda()
        #paddingRowK = torch.zeros((B, C, 1, W))
        paddingColumnK = torch.zeros((B, C, H, 1)).cuda()
    
        #afterPaddingQ1 = torch.cat((q, paddingRowQ),2)
        afterPaddingQ = torch.cat((paddingColumnQ, q),3).cuda()
        
        #afterPaddingK1 = torch.cat((paddingRowK, k),2)
        afterPaddingK = torch.cat((k, paddingColumnK),3).cuda()      
        
        out = afterPaddingQ*afterPaddingK
        out = out[:, :, :, 1:W+1]
        out = torch.sum(out, dim = 1) 
        return out
    def kDown(self, q, k):

        B, C, H, W = q.shape

        paddingRowQ = torch.zeros((B, C, 1, W)).cuda()
        #paddingColumnQ = torch.zeros((B, C, H+1, 1))
        paddingRowK = torch.zeros((B, C, 1, W)).cuda()
        #paddingColumnK = torch.zeros((B, C, H+1, 1))
    
        afterPaddingQ = torch.cat((paddingRowQ, q), 2).cuda()
        #afterPaddingQ2 = torch.cat((afterPaddingQ1, paddingColumnQ),3)
        
        afterPaddingK = torch.cat((k, paddingRowK), 2).cuda()
        #afterPaddingK2 = torch.cat((paddingColumnK, afterPaddingK1),3)      
        
        out = afterPaddingQ*afterPaddingK
        out = out[:,:,1:H+1, :]
        out = torch.sum(out, dim = 1) 
        return out        
    def kCenter(self, q, k):
        return torch.sum(q * k, dim =1)