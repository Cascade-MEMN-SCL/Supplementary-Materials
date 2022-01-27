
import torch
import torch.nn as nn
from utils.utils import *
from utils.train_options import TrainOptions
import cv2 
import numpy as np
#from data import numpy2cuda
def unit_postprocessing(unit, vid_size=None):
    unit = unit.squeeze()
    unit = unit.cpu().detach().numpy()
    unit = np.clip(unit, -1, 1)
    unit = np.round((np.transpose(unit, (1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
    if unit.shape[:2][::-1] != vid_size and vid_size is not None:
         unit = cv2.resize(unit, (256,256), interpolation=cv2.COLOR_BGR2RGB)
    return unit

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class Conv2D_activa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride,
            padding=0, dilation=1, activation='relu'
    ):
        super(Conv2D_activa, self).__init__()
        self.padding = padding
        if self.padding:
            self.pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            dilation=dilation, bias=None
        )
        self.activation = activation
        if activation == 'relu':
            self.activation = nn.ReLU()

    def forward(self, x):
        if self.padding:
            x = self.pad(x)
        x = self.conv2d(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, dim_intermediate=32, ks=3, s=1):
        super(ResBlk, self).__init__()
        p = (ks - 1) // 2
        self.cba_1 = Conv2D_activa(dim_in, dim_intermediate, ks, s, p, activation='relu')
        self.cba_2 = Conv2D_activa(dim_intermediate, dim_out, ks, s, p, activation=None)

    def forward(self, x):
        y = self.cba_1(x)
        y = self.cba_2(y)
        return y + x


def _repeat_blocks(block, dim_in, dim_out, num_blocks, dim_intermediate=32, ks=3, s=1):
    blocks = []
    for idx_block in range(num_blocks):#num_blocks=3
        if idx_block == 0:
            blocks.append(block(dim_in, dim_out, dim_intermediate=dim_intermediate, ks=ks, s=s))
            #ResBlk(32,32,32,3,1)
        else:
            blocks.append(block(dim_out, dim_out, dim_intermediate=dim_intermediate, ks=ks, s=s))
            #ResBlk(32,32,32,3,1)
    return nn.Sequential(*blocks)


class Encoder(nn.Module):
    def __init__(
            self, dim_in=3, dim_out=32, num_resblk=3,
            use_texture_conv=True, use_motion_conv=True, texture_downsample=True,
            num_resblk_texture=2, num_resblk_motion=2
    ):
        super(Encoder, self).__init__()
        self.use_texture_conv, self.use_motion_conv = use_texture_conv, use_motion_conv

        self.cba_1 = Conv2D_activa(dim_in, 16, 7, 1, 3, activation='relu')
        self.cba_2 = Conv2D_activa(16, 32, 3, 2, 1, activation='relu')

        self.resblks = _repeat_blocks(ResBlk, 32, 32, num_resblk)

        # texture representation
        if self.use_texture_conv:
            self.texture_cba = Conv2D_activa(
                32, 32, 3, (2 if texture_downsample else 1), 1,
                activation='relu'
            )#conv2D_active(32,32,3,2,1,activation='relu')
        self.texture_resblks = _repeat_blocks(ResBlk, 32, dim_out, num_resblk_texture)

        # motion representation
        if self.use_motion_conv:
            self.motion_cba = Conv2D_activa(32, 32, 3, 1, 1, activation='relu')
        self.motion_resblks = _repeat_blocks(ResBlk, 32, dim_out, num_resblk_motion)

    def forward(self, x):
        x = self.cba_1(x)#padding+conv2d+activation    16
        x = self.cba_2(x)#padding+conv2d+activation    32
        x = self.resblks(x)#conv2d   32

        if self.use_texture_conv:
            texture = self.texture_cba(x)  #32
            texture = self.texture_resblks(texture) #32
        else:
            texture = self.texture_resblks(x)

        if self.use_motion_conv:
            motion = self.motion_cba(x) #32
            motion = self.motion_resblks(motion) #32
        else:
            motion = self.motion_resblks(x)

        return texture, motion


class Decoder(nn.Module):
    def __init__(self, dim_in=32, dim_out=3, num_resblk=9, texture_downsample=True):
        super(Decoder, self).__init__()
        self.texture_downsample = texture_downsample

        if self.texture_downsample:
            self.texture_up = nn.UpsamplingNearest2d(scale_factor=2)
            # self.texture_cba = Conv2D_activa(dim_in, 32, 3, 1, 1, activation='relu')

        self.resblks = _repeat_blocks(ResBlk, 64, 64, num_resblk, dim_intermediate=64)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.cba_1 = Conv2D_activa(64, 32, 3, 1, 1, activation='relu')
        self.cba_2 = Conv2D_activa(32, dim_out, 7, 1, 3, activation=None)

    def forward(self, texture, motion):
        if self.texture_downsample:
            texture = self.texture_up(texture)
        if motion.shape != texture.shape:
            texture = nn.functional.interpolate(texture, size=motion.shape[-2:])
        x = torch.cat([texture, motion], 1)

        x = self.resblks(x)

        x = self.up(x)
        x = self.cba_1(x)
        x = self.cba_2(x)

        return x


class Manipulator(nn.Module):
    def __init__(self):
        super(Manipulator, self).__init__()
        self.g = Conv2D_activa(32, 32, 3, 1, 1, activation='relu')
        self.h_conv = Conv2D_activa(32, 32, 3, 1, 1, activation=None)
        self.h_resblk = ResBlk(32, 32)

    def forward(self, motion_A, motion_B, amp_factor):
        motion = motion_B - motion_A
        motion_delta = self.g(motion) * amp_factor
        motion_delta = self.h_conv(motion_delta)
        motion_delta = self.h_resblk(motion_delta)
        motion_mag = motion_B + motion_delta
        return motion_mag


class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.encoder = Encoder(dim_in=3*1)
        self.manipulator = Manipulator()
        self.decoder = Decoder(dim_out=3*1)

    def forward(self, batch_A, batch_B, batch_C, batch_M, amp_factor, mode='train'):
        if mode == 'train':
            texture_A, motion_A = self.encoder(batch_A)
            texture_B, motion_B = self.encoder(batch_B)
            texture_C, motion_C = self.encoder(batch_C)
            texture_M, motion_M = self.encoder(batch_M)
            motion_mag = self.manipulator(motion_A, motion_B, amp_factor)
            y_hat = self.decoder(texture_B, motion_mag)
            texture_AC = [texture_A, texture_C]
            motion_BC = [motion_B, motion_C]
            texture_BM = [texture_B, texture_M]
            return y_hat, texture_AC, texture_BM, motion_BC
        elif mode == 'evaluate':
            texture_A, motion_A = self.encoder(batch_A)
            texture_B, motion_B = self.encoder(batch_B)
            motion_mag = self.manipulator(motion_A, motion_B, amp_factor)
            y_hat = self.decoder(texture_B, motion_mag)
            return y_hat, motion_mag, texture_A, texture_B

class MagNetextension(nn.Module):
    def __init__(self,args):
        super(MagNetextension, self).__init__()
        
        student1 = MagNet()
        load_SC_model(student1,args)
        self.student1= student1
       
        student2 = MagNet()
        load_SD_model(student2,args)
        self.student2= student2
        
        
        student3 = MagNet()
        load_SE_model(student3,args)
        self.student3= student3
        
        
#        student4 = MagNet()
#        load_SF_model(student4,args)
#        self.student4= student4
        
        teacher =MagNet().cuda() 
        load_T_model(teacher,args)
        print_model_parm_nums(teacher, 'teacher_model')
        self.teacher = teacher
        

    def forward(self, batch_A, batch_B,labels,step,mode='train'):
        if mode == 'train':
            self.labels=labels
            self.batch_A=batch_A
            self.batch_B=batch_B
            self.motionMagT=[0,0,0]#teacher magnification shape
            self.textureAT=[0,0,0]#teacher A texture
            self.textureBT=[0,0,0]#teacher B texture
            self.preds_T=[0,0,0]
            
            self.motionMag=[0,0,0]#student first handshake magnification shape
            self.textureA=[0,0,0]#student A first handshake texture
            self.textureB=[0,0,0]#student B first handshake texture
            self.preds_C=[0,0,0]#student first handshake output
            
            self.motionMag_back=[0,0,0]#student second handshake magnification shape
            self.textureA_back=[0,0,0]#student A second handshake texture
            self.textureB_back=[0,0,0]#student B second handshake texture
            self.preds_C_back=[0,0,0]#student second handshake output
            
            self.motionMag_twice=[0,0,0]#student third handshake magnification shape
            self.textureB_twice=[0,0,0]#student A third handshake texture
            self.textureA_twice=[0,0,0]#student B third handshake texture
            self.preds_C1=[0,0,0,0,0]#student third handshake output
            
            self.preds_C1[0]=self.batch_A
            self.preds_C1[1]=self.batch_B
            
            #use KD condition,input A B output C
            factor =10.0
            with torch.no_grad():
                 self.preds_T[0], self.motionMagT[0], self.textureAT[0], self.textureBT[0]= self.teacher.eval()(self.preds_C1[0], self.preds_C1[1],0,0,amp_factor=factor, mode='evaluate')
            self.preds_C[0], self.motionMag[0], self.textureA[0], self.textureB[0] = self.student1.train()(self.preds_C1[0], self.preds_C1[1],0,0,amp_factor=factor, mode='evaluate')    

            factor=10.0
            with torch.no_grad():
                 self.preds_T[1], self.motionMagT[1], self.textureAT[1], self.textureBT[1]= self.teacher.eval()(self.preds_C1[1], self.labels[0],0,0,amp_factor=factor, mode='evaluate')
            self.preds_C[1], self.motionMag[1], self.textureA[1], self.textureB[1] = self.student2.train()(self.preds_C1[1], self.preds_C[0],0,0,amp_factor=factor, mode='evaluate')    

            factor=10.0
            with torch.no_grad():
                 self.preds_T[2], self.motionMagT[2], self.textureAT[2], self.textureBT[2]= self.teacher.eval()(self.labels[0], self.labels[1],0,0,amp_factor=factor, mode='evaluate')
            self.preds_C[2], self.motionMag[2], self.textureA[2], self.textureB[2] = self.student3.train()(self.preds_C[0], self.preds_C[1],0,0,amp_factor=factor, mode='evaluate')    

            
            return self.preds_C, self.preds_C_back,self.textureAT,self.textureBT,self.motionMagT,self.textureA, self.textureB,self.motionMag, self.preds_C1
        elif mode == 'evaluate':
            self.batch_A=batch_A
            self.batch_B=batch_B
            self.motionMagT=[0,0,0]#teacher magnification shape
            self.textureAT=[0,0,0]#teacher A texture
            self.textureBT=[0,0,0]#teacher B texture
            
            self.motionMag=[0,0,0]#student first handshake magnification shape
            self.textureA=[0,0,0]#student A first handshake texture
            self.textureB=[0,0,0]#student B first handshake texture
            self.preds_C=[0,0,0]#student first handshake output
            
            self.motionMag_back=[0,0,0]#student second handshake magnification shape
            self.textureA_back=[0,0,0]#student A second handshake texture
            self.textureB_back=[0,0,0]#student B second handshake texture
            self.preds_C_back=[0,0,0]#student second handshake output
            
            self.motionMag_twice=[0,0,0]#student third handshake magnification shape
            self.textureB_twice=[0,0,0]#student A third handshake texture
            self.textureA_twice=[0,0,0]#student B third handshake texture
            self.preds_C1=[0,0,0,0,0]#student third handshake output
            
            self.preds_C1[0]=self.batch_A
            self.preds_C1[1]=self.batch_B
            
            #use KD condition,input A B output C
            factor = 10.0
            
            self.preds_C[0], self.motionMag[0], self.textureA[0], self.textureB[0] = self.student1.eval()(self.preds_C1[0], self.preds_C1[1],0,0,amp_factor=factor, mode='evaluate')    
            #self.preds_C_back[0], self.motionMag_back[0], self.textureA_back[0], self.textureB_back[0] = self.student1.eval()(self.preds_C[0], self.preds_C1[1],0,0,amp_factor=1/factor, mode='evaluate')    
           # self.preds_C1[2], self.motionMag_twice[0], self.textureA_twice[0], self.textureB_twice[0] = self.student1.eval()(self.preds_C_back[0],self.preds_C1[1],0,0,amp_factor=factor, mode='evaluate') 
            #not use KD,input B C output D
            #print(self.student1.state_dict())
            path='result_middle/C'
            if not os.path.exists(path):
                os.makedirs(path)
#            frame = unit_postprocessing(self.preds_C1[0])
#            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#            cv2.imwrite('result_middle/C/{}1.jpg'.format(step), frame)
#            frame = unit_postprocessing(self.preds_C1[1])
#            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#            cv2.imwrite('result_middle/C/{}2.jpg'.format(step), frame)
            frame = unit_postprocessing(self.preds_C[0])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('result_middle/C/{}3.jpg'.format(step), frame)
            #print(self.student1.state_dict())
           
            factor=10.0
            self.preds_C[1], self.motionMag[1], self.textureA[1], self.textureB[1] = self.student2.eval()(self.preds_C1[1], self.preds_C[0],0,0,amp_factor=factor, mode='evaluate')    
            #self.preds_C_back[1], self.motionMag_back[1], self.textureA_back[1], self.textureB_back[1] = self.student2.eval()(self.preds_C[1], self.preds_C1[2],0,0,amp_factor=1/factor, mode='evaluate')    
            #self.preds_C1[3], self.motionMag_twice[1], self.textureA_twice[1], self.textureB_twice[1] = self.student2.eval()(self.preds_C_back[1], self.preds_C1[2],0,0,amp_factor=factor, mode='evaluate')
          
            path='result_middle/D'
            if not os.path.exists(path):
                os.makedirs(path)
#            frame = unit_postprocessing(self.preds_C1[1])
#            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#            cv2.imwrite('result_middle/D/{}1.jpg'.format(step), frame)
#            frame = unit_postprocessing(self.preds_C[0])
#            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#            cv2.imwrite('result_middle/D/{}2.jpg'.format(step), frame)
            frame = unit_postprocessing(self.preds_C[1])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('result_middle/D/{}3.jpg'.format(step), frame)
            
            self.preds_C[2], self.motionMag[2], self.textureA[2], self.textureB[2] = self.student3.eval()(self.preds_C[0], self.preds_C[1],0,0,amp_factor=factor, mode='evaluate')    
            #self.preds_C_back[2], self.motionMag_back[2], self.textureA_back[2], self.textureB_back[2] = self.student3.eval()(self.preds_C[2], self.preds_C1[3],0,0,amp_factor=1/factor, mode='evaluate')    
            #self.preds_C1[4], self.motionMag_twice[2], self.textureA_twice[2], self.textureB_twice[2] = self.student3.eval()(self.preds_C_back[2], self.preds_C1[3],0,0,amp_factor=factor, mode='evaluate')
            
            path='result_middle/E'
            if not os.path.exists(path):
                os.makedirs(path)
#            frame = unit_postprocessing(self.preds_C[0])
#            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#            cv2.imwrite('result_middle/E/{}1.jpg'.format(step), frame)
#            frame = unit_postprocessing(self.preds_C[1])
#            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#            cv2.imwrite('result_middle/E/{}2.jpg'.format(step), frame)
            frame = unit_postprocessing(self.preds_C[2])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite('result_middle/E/{}3.jpg'.format(step), frame)
            
            #self.preds_C[3], self.motionMag[3], self.textureA[3], self.textureB[3] = self.student4.eval()(self.preds_C[1], self.preds_C[2],0,0,amp_factor=factor, mode='evaluate')    
            #self.preds_C_back[3], self.motionMag_back[3], self.textureA_back[3], self.textureB_back[3] = self.student4.eval()(self.preds_C[3], self.preds_C1[4],0,0,amp_factor=1/factor, mode='evaluate')    
            #self.preds_C1[5], self.motionMag_twice[3], self.textureA_twice[3], self.textureB_twice[3] = self.student4.eval()(self.preds_C_back[3], self.preds_C1[4],0,0,amp_factor=factor, mode='evaluate')
            
            # path='result_middle/F'
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # frame = unit_postprocessing(self.preds_C[1])
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('result_middle/F/{}1.jpg'.format(step), frame)
            # frame = unit_postprocessing(self.preds_C[2])
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('result_middle/F/{}2.jpg'.format(step), frame)
            # frame = unit_postprocessing(self.preds_C[3])
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('result_middle/F/{}3.jpg'.format(step), frame)
            
            return self.preds_C
            
def main():
    model = MagNet()
    print('model:\n', model)


if __name__ == '__main__':
    main()
