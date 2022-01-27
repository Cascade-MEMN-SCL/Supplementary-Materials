# -*- coding: utf-8 -*-


#import torchvision
import torch
import torchvision.transforms as transforms
import argparse
#from torchvision.transforms import ToPILImage
#from models import Resnet50
from models import PixelTransformerResnet, Resnet50
#from models import Resnet50
import copy
import time
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import os
import cv2
import torch.utils.data as Data
from dataset import CelebADataset
#from myutils import plot_train_and_test_result



parser = argparse.ArgumentParser(description='resnet for micro-expression')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--classes', type=int, default=3,
                    help='Number of classifications (default: 3)')
parser.add_argument('--heads', type=int, default=8,
                    help='Number of heads of the mutli-head self-attention (default: 8)')

args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


TN_Fold = {}
TN_Fold['samm'] = np.zeros(3, dtype=int)
TN_Fold['smic'] = np.zeros(3, dtype=int)
TN_Fold['casme2'] = np.zeros(3, dtype=int)
TN_Fold['total'] = np.zeros(3, dtype=int)

TP_Fold = {}
TP_Fold['samm'] = np.zeros(3, dtype=int)
TP_Fold['smic'] = np.zeros(3, dtype=int)
TP_Fold['casme2'] = np.zeros(3, dtype=int)
TP_Fold['total'] = np.zeros(3, dtype=int)

FP_Fold = {}
FP_Fold['samm'] = np.zeros(3, dtype=int)
FP_Fold['smic'] = np.zeros(3, dtype=int)
FP_Fold['casme2'] = np.zeros(3, dtype=int)
FP_Fold['total'] = np.zeros(3, dtype=int)

FN_Fold = {}
FN_Fold['samm'] = np.zeros(3, dtype=int)
FN_Fold['smic'] = np.zeros(3, dtype=int)
FN_Fold['casme2'] = np.zeros(3, dtype=int)
FN_Fold['total'] = np.zeros(3, dtype=int)

loss_fun = torch.nn.CrossEntropyLoss()

for ai in range(1, 69):
    print(args)
    print('this is the ' + str(ai) + ' th of the loso' )
    
    trainlosses = []
    testlosses = []    
    batch_size = args.batch_size
    n_classes = args.classes
    epochs = args.epochs
    dropout = args.dropout
    n_heads = args.heads
    
    
    if ai<10:
       train_name='train0'+str(ai)
    else:
       train_name='train'+str(ai)
    train_path=os.path.join('./dataset_crop_center/amplified/train',train_name)
    images1=os.listdir(train_path)
    train=[]
    train_dict={}
    train_dict_path=[]
    for i in range(len(images1)):
        path=os.path.join(train_path,images1[i])
        image=cv2.imread(path)
        image=cv2.resize(image,(200,200))
        image.astype(float)
        train.append(image)
        
        # train_dict_path.append(path.split("\\")[-1])
        # train_dict[path.split("\\")[-1]]=image
    
    train_data1= torch.FloatTensor(np.array(train))/256
    train_data1=train_data1.permute(0,3, 1, 2)
    

    if ai<10:
       test_name='test0'+str(ai)
    else:
       test_name='test'+str(ai)
    test_path=os.path.join('./dataset_crop_center/amplified/test',test_name)
    images2=os.listdir(test_path)
    test=[]
    test_dict={}
    test_dict_path=[]
    for i in range(len(images2)):
        path=os.path.join(test_path,images2[i])
        image=cv2.imread(path)
        image=cv2.resize(image,(200,200))
        image.astype(float)
        test.append(image)
        
        # test_dict_path.append(path.split("\\")[-1])
        # test_dict[path.split("\\")[-1]]=image
        
    test_data= torch.FloatTensor(np.array(test))/256
    test_data=test_data.permute(0,3, 1, 2)
    

    if ai<10:
       train_label_name='train0'+str(ai)+'.txt'
    else:
       train_label_name='train'+str(ai)+'.txt'
    train_label_path='./dataset_crop_center/amplified/train_label'+'/'+train_label_name
    f=open(train_label_path)
    arr=[]
    train_dict_label={}
    for lines in f.readlines():
        lines=lines.strip('\n')
        arr.append(int(lines)) 
        
        # train_dict_label[lines.split(" ")[0]]=int(lines.split(" ")[-1])
   
    f.close()
    
   
    
    train_label = torch.LongTensor(np.array(arr))
    

    if ai<10:
       test_label_name='test0'+str(ai)+'.txt'
    else:
       test_label_name='test'+str(ai)+'.txt'
    test_label_path='./dataset_crop_center/amplified/test_label'+'/'+test_label_name
    f1=open(test_label_path)
    test_arr=[]
    test_dict_label={}
    for lines in f1.readlines():
        lines=lines.strip('\n')
        test_arr.append(int(lines))
        
        # test_dict_label[lines.split(" ")[0]]=int(lines.split(" ")[-1])
        
    f1.close()
    test_label = torch.LongTensor(np.array(test_arr))
    print(test_label_path)
    print(train_data1.shape)
    print(train_label.shape)
    

    train_data = Data.TensorDataset(train_data1,train_label)
    test_data = Data.TensorDataset(test_data,test_label)
 
    # train_data=CelebADataset(train_dict_path,train_dict,train_dict_label)
    # test_data=CelebADataset(test_dict_path,test_dict,test_dict_label)
    
    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False, num_workers=0)
    
    # net = Resnet50(mid_dim = 256, last_dim = n_classes, dropout = dropout)
    net = PixelTransformerResnet(mid_dim = 256, last_dim =3, dropout = dropout, n_heads = n_heads)
    '''
    print(net)
    
    for k, v in net.named_parameters():
        print(k)
        #print(v)
        print(v.requires_grad)
    '''  
    if args.cuda:
        net.cuda()
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
  
    lr = args.lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(net.state_dict())
    best_pred = []
    best_target = []
    Epoch_F1_score = 0.0
    FP_Epoch = 0.0
    FN_Epoch = 0.0
    TP_Epoch = 0.0
    TN_Epoch = 0.0
    
    def train(epoch):
        fp.write(('-' * 100) + "\n")
        train_loss = 0
        correct_train = 0
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda: data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss = loss_fun(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss
            
            pred_train = output.data.max(1, keepdim=True)[1]
            correct_train += pred_train.eq(target.data.view_as(pred_train)).cpu().sum()
            
            if batch_idx > 0 and (batch_idx + 1) % len(train_loader) == 0:
                print('Train Epoch: {} \tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)'.format(
                    epoch, train_loss.item()/len(train_loader), correct_train, len(train_loader.dataset),
                100. * correct_train / len(train_loader.dataset)))
                
                fp.write('Train Epoch: {} \tLoss: {:.6f}\t Accuracy: {}/{} ({:.0f}%)'.format(
                    epoch, train_loss.item()/len(train_loader), correct_train, len(train_loader.dataset),
                100. * correct_train / len(train_loader.dataset))+"\n")
                                         
                taloss = train_loss.item()/len(train_loader)
                train_loss = 0
            # return taloss
    def test():
        net.eval()
        test_loss = 0
        correct = 0
        all_target_list = []
        all_pred_list = []
        global best_acc
        global best_model_wts
        global best_pred
        global best_target
        #global Epoch_accuracy 
        global Epoch_F1_score
        global FP_Epoch
        global FN_Epoch
        global TP_Epoch 
        global TN_Epoch        
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = net(data)
                test_loss += loss_fun(output, target).item()
                
                pred = output.data.max(1, keepdim = True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                
                target_npy = target.cpu().numpy()
                target_list = target_npy.tolist()
                all_target_list.extend(target_list) 
                pred_npy = pred.cpu().numpy().reshape(len(target_list))
                pred_list = pred_npy.tolist()
                all_pred_list.extend(pred_list)
            test_loss /= len(test_loader)
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))
            print('test label: {}'.format(all_target_list))
            print('pred label: {}\n'.format(all_pred_list))
            fp.write(('-' * 10)+"\n")
            fp.write(('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset))))
                
            fp.write('test label: {}'.format(all_target_list))
            fp.write('pred label: {}\n'.format(all_pred_list))
            epoch_acc = 100. * correct / len(test_loader.dataset)
            
            matrix = confusion_matrix(all_target_list, all_pred_list, labels=[0, 1, 2])
            FP = matrix.sum(axis=0) - np.diag(matrix)
            FN = matrix.sum(axis=1) - np.diag(matrix)
            TP = np.diag(matrix)
            TN = matrix.sum() - (FP + FN + TP)

            f1_s = np.ones([3])
            deno = (2 * TP + FP + FN)
            for f in range(3):
                if deno[f] != 0:
                    f1_s[f] = (2 * TP[f]) / (2 * TP[f] + FP[f] + FN[f])
                else:
                    f1_s[f] = 1

            f1 = np.mean(f1_s)

            

            if f1 > Epoch_F1_score:
                Epoch_F1_score = f1
                #Epoch_accuracy = epoch_acc
                FP_Epoch = FP
                FN_Epoch = FN
                TP_Epoch = TP
                TN_Epoch = TN
                #sample_file = temp_file
                best_model_wts = copy.deepcopy(net.state_dict())
                best_acc = epoch_acc
                best_pred = all_pred_list
                best_target = all_target_list
            elif f1 == Epoch_F1_score:
                if epoch_acc >= best_acc:
                    Epoch_F1_score = f1
                    #Epoch_accuracy = epoch_acc
                    FP_Epoch = FP
                    FN_Epoch = FN
                    TP_Epoch = TP
                    TN_Epoch = TN
                    #sample_file = temp_file
                    best_model_wts = copy.deepcopy(net.state_dict())
                    best_acc = epoch_acc
                    best_pred = all_pred_list
                    best_target = all_target_list
                    
        return best_model_wts, best_acc, best_pred, best_target, FP_Epoch, FN_Epoch, TP_Epoch, TN_Epoch, test_loss
    if __name__ =="__main__":
        since = time.time()
        for epoch in range(1, epochs+1):
            global fp
            fp = open('out/txt/result'+str(ai)+'.txt', 'a+')
            train(epoch)
            best_model_wts, best_acc, best_pre, best_tar,FP_Epoch,FN_Epoch,TP_Epoch,TN_Epoch,testloss =  test()
            
            if best_acc == 100:
                break
            if epoch % 30 == 0:
                lr /= 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        if 1 <=ai <= 24:
            TP_Fold['casme2'] += TP_Epoch
            TN_Fold['casme2'] += TN_Epoch
            FN_Fold['casme2'] += FN_Epoch
            FP_Fold['casme2'] += FP_Epoch
        elif 25 <=ai <= 52:
            TP_Fold['samm'] += TP_Epoch
            TN_Fold['samm'] += TN_Epoch
            FN_Fold['samm'] += FN_Epoch
            FP_Fold['samm'] += FP_Epoch
        elif 53 <=ai <= 68:
            TP_Fold['smic'] += TP_Epoch
            TN_Fold['smic'] += TN_Epoch
            FN_Fold['smic'] += FN_Epoch
            FP_Fold['smic'] += FP_Epoch
            
        TP_Fold['total'] += TP_Epoch
        TN_Fold['total'] += TN_Epoch
        FN_Fold['total'] += FN_Epoch
        FP_Fold['total'] += FP_Epoch 
        
        net.load_state_dict(best_model_wts)
        torch.save(net.state_dict() , 'out/pth/newbest'+str(ai)+'.pth')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))
        fp.write('\nBest test Acc: {:4f}'.format(best_acc))
        fp.close()
        fl = open('out/label/result_label'+str(ai)+'.txt', 'a+')
        fl.write('Best  pred: {}\n'.format(best_pre))
        fl.write('    target: {}'.format(best_tar))
        fl.close()

        
F1_Score = {}
F1_Score['samm'] = (2 * TP_Fold['samm']) / (2 * TP_Fold['samm'] + FP_Fold['samm'] + FN_Fold['samm'])
F1_Score['smic'] = (2 * TP_Fold['smic']) / (2 * TP_Fold['smic'] + FP_Fold['smic'] + FN_Fold['smic'])
F1_Score['casme2'] = (2 * TP_Fold['casme2']) / (2 * TP_Fold['casme2'] + FP_Fold['casme2'] + FN_Fold['casme2'])
F1_Score['total'] = (2 * TP_Fold['total']) / (2 * TP_Fold['total'] + FP_Fold['total'] + FN_Fold['total'])

Recall_Score = {}
Recall_Score['samm'] = TP_Fold['samm'] / (TP_Fold['samm'] + FN_Fold['samm'])
Recall_Score['smic'] = TP_Fold['smic'] / (TP_Fold['smic'] + FN_Fold['smic'])
Recall_Score['casme2'] = TP_Fold['casme2'] / (TP_Fold['casme2'] + FN_Fold['casme2'])
Recall_Score['total'] = TP_Fold['total'] / (TP_Fold['total'] + FN_Fold['total'])

#Total_accuracy = Fold_accuracy / FOLD

Total_F1_Score = {}
Total_F1_Score['samm'] = np.mean(F1_Score['samm'])
Total_F1_Score['smic'] = np.mean(F1_Score['smic'])
Total_F1_Score['casme2'] = np.mean(F1_Score['casme2'])
Total_F1_Score['total'] = np.mean(F1_Score['total'])

Total_Recall = {}
Total_Recall['samm'] = np.mean(Recall_Score['samm'])
Total_Recall['smic'] = np.mean(Recall_Score['smic'])
Total_Recall['casme2'] = np.mean(Recall_Score['casme2'])
Total_Recall['total'] = np.mean(Recall_Score['total'])

print('\nFold F1 score SAMM: ', Total_F1_Score['samm'])
print('Fold F1 score SMIC: ', Total_F1_Score['smic'])
print('Fold F1 score CASMEII: ', Total_F1_Score['casme2'])
print('Fold F1 score: ', Total_F1_Score['total'])

print('Fold Recall score SAMM: ', Total_Recall['samm'])
print('Fold Recall score SMIC: ', Total_Recall['smic'])
print('Fold Recall score CASMEII: ', Total_Recall['casme2'])
print('Fold Recall score: ', Total_Recall['total'])

jieguo = open('out/uf1_uar.txt', 'a+')
jieguo.write('\nFold F1 score SAMM: {} \n'.format(Total_F1_Score['samm']))
jieguo.write('Fold F1 score SMIC: {} \n'.format(Total_F1_Score['smic']))
jieguo.write('Fold F1 score CASMEII: {} \n'.format(Total_F1_Score['casme2']))
jieguo.write('Fold F1 score: {} \n'.format(Total_F1_Score['total']))

jieguo.write('Fold Recall score SAMM: {} \n'.format(Total_Recall['samm']))
jieguo.write('Fold Recall score SMIC: {} \n'.format(Total_Recall['smic']))
jieguo.write('Fold Recall score CASMEII: {} \n'.format(Total_Recall['casme2']))
jieguo.write('Fold Recall score: {} \n'.format(Total_Recall['total']))
jieguo.close()