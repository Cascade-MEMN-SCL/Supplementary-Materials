import os
import numpy as np
import torch
import cv2
from skimage import io
from utils.train_options import TrainOptions
def load_unit(path):
    file_suffix = path.split('.')[-1].lower()
    if file_suffix in ['jpg', 'png']:
        try:
            unit = cv2.cvtColor(io.imread(path).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print('{} load exception:\n'.format(path), e)
    else:
        print('Unsupported file type.')
        return None
    unit = cv2.resize(unit, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    try:
        unit = unit.astype(np.float32) / 127.5 - 1.0
    except Exception as e:
        print('EX:', e, unit.shape, unit.dtype)

    unit = np.transpose(unit, (2, 0, 1))
    return unit

def numpy2cuda(array):
    tensor = torch.from_numpy(np.asarray(array)).float().cuda()
    return tensor
def get_image(path):
        files_A=[]
        paths_A=[]
        files_A = sorted(os.listdir(path), key=lambda x: x.split('.')[0])
        paths_A = [os.path.join(path, file_A) for file_A in files_A]
        return paths_A
class DataGen():
    def __init__(self, args):
        self.args = args
        self.anchor = 0
        self.paths =args.dataset_path
        self.batch_size =args.batch_size
        self.data_len =len(os.listdir( self.paths))
    
    def gen(self, anchor=None):
        batch_A = []
        batch_M = []
        batch_B = []
        batch_C = []
        batch_D = []
        if anchor is None:
            anchor = self.anchor
        for _ in range(self.batch_size):
            unit_A = load_unit(get_image(self.paths)[self.anchor])
            unit_M = load_unit(get_image(self.paths.replace('frameA', 'amplified'))[self.anchor])
            unit_B = load_unit(get_image(self.paths.replace('frameA', 'frameB'))[self.anchor])
            unit_C = load_unit(get_image(self.paths.replace('frameA', 'frameC'))[self.anchor])
            unit_D = load_unit(get_image(self.paths.replace('frameA', 'frameD'))[self.anchor])
            #unit_E = load_unit(get_image(self.paths.replace('frameA', 'frameE'))[self.anchor])
            batch_M.append(unit_M)
            batch_A.append(unit_A)
            batch_B.append(unit_B)
            batch_C.append(unit_C)
            batch_D.append(unit_D)
            #batch_E.append(unit_E)
            self.anchor = (self.anchor+1)% self.data_len
        batch_A = numpy2cuda(batch_A)
        batch_M = numpy2cuda(batch_M)
        batch_B = numpy2cuda(batch_B)
        batch_C = numpy2cuda(batch_C)
        batch_D = numpy2cuda(batch_D)

       # batch_E = numpy2cuda(batch_E)
        return batch_A, batch_B, batch_C, batch_D, batch_M
class test_DataGen():
    def __init__(self, args):
        self.args = args
        self.anchor = 0
        self.paths =args.test_dataset_path
        self.batch_size =args.batch_size
        self.data_len =len(os.listdir( self.paths))
    
    def gen(self, anchor=None):
        batch_A = []
        batch_B = []
        if anchor is None:
            anchor = self.anchor
        for _ in range(self.batch_size):
            unit_A = load_unit(get_image(self.paths)[self.anchor])
            unit_B = load_unit(get_image(self.paths.replace('frameA', 'frameB'))[self.anchor])
            batch_A.append(unit_A)
            batch_B.append(unit_B)
            self.anchor = (self.anchor+1)% self.data_len
        batch_A = numpy2cuda(batch_A)
        batch_B = numpy2cuda(batch_B)
        return batch_A, batch_B

