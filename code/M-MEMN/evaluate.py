# -*- coding: utf-8 -*-
import torch
import numpy as np
import cv2
from utils.train_options import TrainOptions
import os
def unit_postprocessing(unit, vid_size=None):
    unit = unit.squeeze()
    unit = unit.cpu().detach().numpy()
    unit = np.clip(unit, -1, 1)
    unit = np.round((np.transpose(unit, (1, 2, 0)) + 1.0) * 127.5).astype(np.uint8)
    if unit.shape[:2][::-1] != vid_size and vid_size is not None:
         unit = cv2.resize(unit, (256,256), interpolation=cv2.COLOR_BGR2RGB)
    return unit

def evalute_model(model, loader,args,epoch):
    ep = ''
    weights_file = "teacher_LVMM.pth"
    weights_path = os.path.join(args.save_dir, weights_file)
    #ep = int(weights_path.split('epoch')[-1].split('_')[0])
    ep=epoch
    model_test =model
    model_test.eval()
    testset = 'amplified'
    dir_results = args.save_dir.replace('weights', 'results')
    if not os.path.exists(dir_results):
        os.makedirs(dir_results)
    
    data_loader = loader
    print('Number of test image couples:', data_loader.data_len)
    files_A=[]
    paths_A=[]
    temp=[]
    files_A = sorted(os.listdir(data_loader.paths), key=lambda x: x.split('.')[0])
    paths_A = [os.path.join(data_loader.paths, file_A) for file_A in files_A]
    vid_size = cv2.imread(paths_A[0]).shape[:2][::-1]
    # Test
    for amp in [10]:
        frames = []
        frames1=[]
        for idx_load in range(0, data_loader.data_len, data_loader.batch_size):
            if (idx_load+1) % 100 == 0:
                print('{}'.format(idx_load+1), end=', ')
            batch_A, batch_B = data_loader.gen()[0:2]#gen_test()
            amp_factor = tensor = torch.from_numpy(np.asarray(amp)).float().cuda()#
            for _ in range(len(batch_A.shape) - len(amp_factor.shape)):
                amp_factor = amp_factor.unsqueeze(-1)
            with torch.no_grad():
                temp2 = model_test(batch_A, batch_B, 0,idx_load, mode='evaluate')
            for i,y_hat in enumerate(temp2[-1]):
                A = unit_postprocessing(batch_A[i], vid_size=vid_size)
                B= unit_postprocessing(batch_B[i], vid_size=vid_size)
                y_hat = unit_postprocessing(y_hat, vid_size=vid_size)

                frames.append(A)
                frames.append(B)
                frames.append(y_hat)
            
                if len(frames) >= data_loader.data_len*3:
                    break
            if len(frames) >= data_loader.data_len*3:
                break
        data_loader = loader
        '''for gen in data_loader.gen()[0]:
            
            frames1.append(gen)
            if len(frames1) >= data_loader.data_len:
                break'''
        #frames =frames#gen_test()
        # Make videos of framesMag
        video_dir = os.path.join('ep{}'.format(ep), testset)
        video_dirC = os.path.join('ep{}C'.format(ep), testset)
        video_dirB= os.path.join('ep{}B'.format(ep), testset)
        video_dirA = os.path.join('ep{}A'.format(ep), testset)
        video_dir=os.path.join(video_dir,'{}'.format(amp))
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        video_dirA=os.path.join(video_dirA,'{}'.format(amp))
        if not os.path.exists(video_dirA):
            os.makedirs(video_dirA)
        video_dirB=os.path.join(video_dirB,'{}'.format(amp))
        if not os.path.exists(video_dirB):
            os.makedirs(video_dirB)
        video_dirC=os.path.join(video_dirC,'{}'.format(amp))
        if not os.path.exists(video_dirC):
            os.makedirs(video_dirC)
        h=0
        k=0
        j=0
        
        for i,frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if((i-(i//3*3))==2):
                h=h+1
                cv2.imwrite(os.path.join(video_dirC, '{}.jpg'.format(h)), frame)
                cv2.putText(frame, 'amp_factor={}'.format(amp), (7, 37),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                
            elif((i-(i//3*3))==1):
                j=j+1
                cv2.imwrite(os.path.join(video_dirB, '{}.jpg'.format(j)), frame)
                cv2.putText(frame, 'amp_factor=B', (7, 37),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            else: 
                k=k+1
                cv2.imwrite(os.path.join(video_dirA, '{}.jpg'.format(k)), frame)   
                cv2.putText(frame, 'amp_factor=A', (7, 37),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(video_dir, '{}.jpg'.format(i)), frame)
        print('{} has been done.'.format(os.path.join(video_dir, '{}_amp{}.avi'.format(testset, amp))))
        print('{} has been done.'.format(os.path.join(video_dirA, '{}_amp{}.avi'.format(testset, amp))))
        print('{} has been done.'.format(os.path.join(video_dirB, '{}_amp{}.avi'.format(testset, amp))))
        print('{} has been done.'.format(os.path.join(video_dirC, '{}_amp{}.avi'.format(testset, amp))))