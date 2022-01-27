import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.train_options import TrainOptions
from networks.kd_model import NetModel
import logging
import warnings
warnings.filterwarnings("ignore")
from dataset.datasets import DataGen,test_DataGen
from evaluate import evalute_model
logging.getLogger().setLevel(logging.INFO)
import os.path
import torch
import random
import numpy as np

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(1111)

args = TrainOptions().initialize()
data_loader=DataGen(args)
test_data_loader=test_DataGen(args)
model = NetModel(args)
for epoch in range(args.start_epoch, args.epoch_nums):
      for step in range(0, data_loader.data_len, data_loader.batch_size):
          model.adjust_learning_rate(args.lr_g, model.G_solver, step)
          model.adjust_learning_rate(args.lr_d, model.D_solver, step)
          batch_A,batch_B,batch_C,batch_D,batch_M = data_loader.gen()
          model.set_input(batch_A,batch_B,batch_C, batch_D, batch_M)
          model.optimize_parameters(step)
          model.print_info(epoch, step)
      path_ckpt ='weights_date'
      if not os.path.exists(path_ckpt):
          os.makedirs(path_ckpt)
      path_ckpt1=os.path.join(path_ckpt,'magnet_epoch{}.pth'.format(epoch))
      torch.save(model.student.state_dict(), path_ckpt1)
evalute_model(model.student, test_data_loader,args,epoch)

