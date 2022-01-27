import argparse
import torch
import time
import logging
import os
from utils.utils import *

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description='knowledge-distillation')

        parser.add_argument('--classes_num', default=3, type=int,metavar='N', help='class num of the dataset')


        parser.add_argument('--D_ckpt_path', default='',type=str, metavar='discriminator ckpt path', help='discriminator ckpt path')
        parser.add_argument("--batch-size", type=int, default=1, help="Number of images sent to the network in one step.")
        parser.add_argument('--start_epoch', default=0, type=int,metavar='start_epoch', help='start_epoch')
        parser.add_argument('--epoch_nums', default=15, type=int,metavar='epoch_nums', help='epoch_nums')
        parser.add_argument('--parallel', default='False', type=str, metavar='parallel', help='attribute of saved name')
       
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--num-steps", type=int, default=40000, help="Number of training steps.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
        parser.add_argument("--snapshot-dir", type=str, default='./snapshots/', help="Where to save snapshots of the model.")
        parser.add_argument("--weight-decay", type=float, default=1.e-4, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--preprocess-GAN-mode", type=int, default=1, help="preprocess-GAN-mode should be tanh or bn")
        parser.add_argument('--D_resume', default=True, type=bool,metavar='is or not use discriminator', help='is or not use discriminator ckpt')
        parser.add_argument("--pi", type=str2bool, default='True', help="is pixel wise loss using or not")
        parser.add_argument("--pa", type=str2bool, default='True', help="is pixel wise loss using or not")
        parser.add_argument("--ho", type=str2bool, default='True', help="is pixel wise loss using or not")
        parser.add_argument("--adv-loss-type", type=str, default='wgan-gp', help="adversarial loss setting")
        parser.add_argument("--imsize-for-adv", type=int, default=65, help="imsize for addv")
        parser.add_argument("--adv-conv-dim", type=int, default=64, help="conv dim in adv")
        parser.add_argument("--lambda-gp", type=float, default=10.0, help="lambda_gp")
        parser.add_argument("--lambda-d", type=float, default=0.1, help="lambda_d")
        parser.add_argument("--lambda-pi", type=float, default=1, help="lambda_pi")
        parser.add_argument('--lambda-pa', default=1, type=float, help='')
        parser.add_argument('--pool-scale', default=0.5, type=float, help='')
        parser.add_argument("--lr-g", type=float, default=1e-3, help="learning rate for G")
        parser.add_argument("--lr-d", type=float, default=4e-4, help="learning rate for D")
        parser.add_argument("--best-mean-IU", type=float, default=0.0, help="learning rate for D")
        parser.add_argument("--save-dir", type=str,default="S-MEMN",help="teacher ckpt path")
        parser.add_argument("--save-dir-student", type=str,default="S-MEMN",help="student ckpt path")
        args = parser.parse_args()
        
        args.dataset_path='./clip_dataset/frameA'
        args.test_dataset_path='./clip_dataset/test_frameA'
        args.save_name = 'save_path'
        args.S_ckpt_path = './ckpt/'+ args.save_name +'/Student'
        args.D_ckpt_path = './ckpt/' + args.save_name +'/Distriminator'
        args.D_att_ckpt_path  = './ckpt/' + args.save_name +'/Att_discriminator'
        args.log_path = './ckpt/log/' + args.save_name

        return args


class TrainOptionsForTest():
    def initialize(self):
        parser = argparse.ArgumentParser(description='knowledge-distillation')
        parser.add_argument("--data-dir", type=str, default='', help="")
        parser.add_argument("--resume-from", type=str, default='', help="")
        args = parser.parse_args()
        for key, val in args._get_kwargs():
            print(key+' : '+str(val)) 
        return args

def main():
    args = TrainOptions().initialize()
    print(args.save_dir_student)

if __name__ == '__main__':
    main()