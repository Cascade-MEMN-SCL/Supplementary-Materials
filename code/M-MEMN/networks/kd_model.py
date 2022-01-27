# -*- coding: utf-8 -*-


import argparse
import logging
import os
import pdb
from torch.autograd import Variable
import os.path as osp
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import resources
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.utils import *
import torch.backends.cudnn as cudnn
from utils.criterion import CriterionDSN, CriterionOhemDSN, CriterionPixelWise, \
    CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionPairWiseforWholeFeatAfterPool
from networks.sagan_models import Discriminator
#from networks.evaluate import evaluate_main
from magnet import MagNet,MagNetextension
import cv2

torch_ver = torch.__version__[:3]

def unit_postprocessing(unit, vid_size=None):
    unit = unit.squeeze()
    unit = unit.cpu().detach().numpy()
    unit = np.clip(unit, -1, 1)
    unit = np.round((np.transpose(unit, (1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
    if unit.shape[:2][::-1] != vid_size and vid_size is not None:
         unit = cv2.resize(unit, (256,256), interpolation=cv2.COLOR_BGR2RGB)
    return unit

class NetModel():
    def __init__(self, args):
        cudnn.enabled = True
        self.args = args
        student = MagNetextension(args).cuda()
        load_S_model(student,args)
        print_model_parm_nums(student, 'student_model')
        self.student= student

        teacher =MagNet().cuda() 
        load_T_model(teacher,args)
        print_model_parm_nums(teacher, 'teacher_model')
        self.teacher = teacher

        D_model = Discriminator(args.preprocess_GAN_mode, args.classes_num, args.batch_size, args.imsize_for_adv, args.adv_conv_dim)
       
        print_model_parm_nums(D_model, 'D_model')
       
        self.D_model=D_model 
        
       
        self.G_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, student.parameters()), 'initial_lr': args.lr_g}], args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)
        self.D_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, D_model.parameters()), 'initial_lr': args.lr_d}], args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

        self.best_mean_IU = args.best_mean_IU
        self.criterion = CriterionDSN().cuda()
        self.criterion_pixel_wise =CriterionPixelWise().cuda()
        self.criterion_pair_wise_for_interfeat = CriterionPairWiseforWholeFeatAfterPool(scale=args.pool_scale, feat_ind=-5).cuda()
        self.criterion_adv = CriterionAdv(args.adv_loss_type).cuda()
        if args.adv_loss_type == 'wgan-gp':
            self.criterion_AdditionalGP = CriterionAdditionalGP(self.D_model, args.lambda_gp).cuda()
        self.criterion_adv_for_G = CriterionAdvForG(args.adv_loss_type).cuda()
            
        self.mc_G_loss = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0
        self.D_loss = 0.0

        cudnn.benchmark = True
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

    def AverageMeter_init(self):
        self.parallel_top1_train = AverageMeter()
        self.top1_train = AverageMeter()

    def set_input(self, A,B,C,D,L):
        self.batch_A =A.cuda()
        self.batch_B =B.cuda()
        self.batch_C =C.cuda()
        self.batch_D =D.cuda()
        #self.batch_E =E.cuda()
        self.batch_axpe = L.cuda()
        self.labels=[0,0,0]
        self.labels[0]=self.batch_C 
        self.labels[1]= self.batch_D 
        #self.labels[2]=self.batch_E
        self.labels[2]=self.batch_axpe
        if torch_ver == "0.3":
            self.batch_A = Variable(A)
            self.batch_B = Variable(B)
            self.batch_C = Variable(C)
            self.batch_D = Variable(D)
            #self.batch_E = Variable(E)
            self.batch_axpe = Variable(L)
            self.labels=[0,0,0]
            self.labels[0]=self.batch_C 
            self.labels[1]= self.batch_D 
            #self.labels[2]=self.batch_E
            self.labels[2]=self.batch_axpe

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr*((1-float(iter)/max_iter)**(power))
            
    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def forward(self,step):
        self.preds_C, self.preds_C_back,self.textureAT,self.textureBT,self.motionMagT,self.textureA, \
        self.textureB,self.motionMag, self.preds_C1=self.student.train()(self.batch_A, self.batch_B,self.labels,step,mode='train')
       
    def student_backward(self):
        args = self.args
        G_loss = 0.0
        
        for j in range(0,3): 
            temp = self.criterion(self.preds_C[j], self.labels[j])#student first handshake loss
#            temp_back = self.criterion(self.preds_C_back[j], self.preds_C1[j])
#            temp_twice = self.criterion(self.preds_C1[j+2], self.labels[j])#student third handshake loss
            self.mc_G_loss = temp.item()#+temp_back.item()+temp_twice.item()
            G_loss = G_loss + 0.1*temp
            if args.pi == True:
                tempMotion = self.criterion(self.motionMag[j], self.motionMagT[j])
                #tempMOtionBack = args.lambda_pi*self.criterion(self.motionMag_twice[j], self.motionMagT[j])
                self.pi_G_loss = tempMotion.item() #+ tempMOtionBack.item()
                G_loss = G_loss + args.lambda_pi*tempMotion #+ tempMOtionBack
            if args.pa == True:
                tempTextureB = self.criterion(self.textureB[j], self.textureBT[j])
                tempTextureA= self.criterion(self.textureA[j], self.textureAT[j])
                #tempTexture_twice = self.criterion(self.textureB_twice[j], self.textureBT[j])
                self.pa_G_loss = tempTextureB.item() + tempTextureA.item()#+tempTexture_twice.item()
                G_loss = G_loss + args.lambda_pa*(tempTextureB + tempTextureA)
       
#        for j in range(1,3): 
#            temp = self.criterion(self.preds_C[j], self.labels[j])#student first handshake loss
##            temp_back = self.criterion(self.preds_C_back[j], self.preds_C1[j])
##            temp_twice = self.criterion(self.preds_C1[j+2], self.labels[j])#student third handshake loss
#            self.mc_G_loss = temp.item()#+temp_back.item()+temp_twice.item()
#            G_loss = G_loss +0.1*temp #+temp_back+temp_twice+temp
#            if args.pi == True:
#                tempMotion = self.criterion(self.motionMag[j], self.motionMagT[j])
#                #tempMOtionBack = args.lambda_pi*self.criterion(self.motionMag_twice[j], self.motionMagT[j])
#                self.pi_G_loss = tempMotion.item() #+ tempMOtionBack.item()
#                G_loss = G_loss + args.lambda_pi*tempMotion #+ tempMOtionBack
#            if args.pa == True:
#                tempTextureB = self.criterion(self.textureB[j], self.textureBT[j])
#                tempTextureA= self.criterion(self.textureA[j], self.textureAT[j])
#                #tempTexture_twice = self.criterion(self.textureB_twice[j], self.textureBT[j])
#                self.pa_G_loss = tempTextureB.item() + tempTextureA.item()#+tempTexture_twice.item()
#                G_loss = G_loss + args.lambda_pa*(tempTextureB + tempTextureA)
        if self.args.ho == True:
            d_out_S = self.D_model(eval(compile('self.preds_C[-1]', '<string>', 'eval')))
            G_loss = G_loss + args.lambda_d*self.criterion_adv_for_G(d_out_S, d_out_S)
        G_loss.backward()
        self.G_loss = G_loss.item()

    def discriminator_backward(self):
        self.D_solver.zero_grad()
        args = self.args
        d_out_T = self.D_model(eval(compile('self.labels[-1].detach()', '<string>', 'eval')))
        d_out_S = self.D_model(eval(compile('self.preds_C[-1].detach()', '<string>', 'eval')))
        d_loss = args.lambda_d*self.criterion_adv(d_out_S, d_out_T)

        if args.adv_loss_type == 'wgan-gp':
            d_loss += args.lambda_d*self.criterion_AdditionalGP(self.preds_C[-1], self.labels[-1])

        d_loss.backward()
        self.D_loss = d_loss.item()
        self.D_solver.step()

    def optimize_parameters(self,step):
        self.forward(step)
        self.G_solver.zero_grad()
        self.student_backward()
        self.G_solver.step() 
        if self.args.ho == True:
            self.discriminator_backward()
   
    def unit_postprocessing(self, unit, vid_size=None):
        unit = unit.squeeze()
        unit = unit.cpu().detach().numpy()
        unit = np.clip(unit, -1, 1)
        unit = np.round((np.transpose(unit, (1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
        if unit.shape[:2][::-1] != vid_size and vid_size is not None:
            unit = cv2.resize(unit, vid_size, interpolation=cv2.INTER_CUBIC)
        return unit
    
    
    def print_info(self, epoch, step):
        logging.info('step:{:5d} G_lr:{:.6f} G_loss:{:.5f}(mc:{:.5f} pixelwise:{:.5f} pairwise:{:.5f}) D_lr:{:.6f} D_loss:{:.5f}'.format(
                        step, self.G_solver.param_groups[-1]['lr'], 
                        self.G_loss, self.mc_G_loss, self.pi_G_loss, self.pa_G_loss, 
                        self.D_solver.param_groups[-1]['lr'], self.D_loss))
        logging.info('step:{:5d}')

    def __del__(self):
        pass

    def save_ckpt(self, epoch, step, mean_IU, IU_array):
        torch.save(self.student.state_dict(),osp.join(self.args.snapshot_dir, 'CS_scenes_'+str(step)+'_'+str(mean_IU)+'.pth'))  



