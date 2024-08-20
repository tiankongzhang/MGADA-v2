import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random

class SamplerMemory(nn.Module):
    def __init__(self, num_classes):
        super(SamplerMemory, self).__init__()
        #label params
        self.class_numbers = num_classes
        
        #memory
        self.memory_total_sz = 10
        self.images_sample_memory = [[None for i in range(self.memory_total_sz)] for idx in range(self.class_numbers)]
        self.targets_sample_memory = [[None for i in range(self.memory_total_sz)] for idx in range(self.class_numbers)]
        
        self.sample_memory_index = np.zeros((self.class_numbers))
        self.sample_memory_loc = np.zeros((self.class_numbers))
        self.sample_memory_ovf = np.zeros((self.class_numbers))
        self.state_memory = False
        self.read_sample_index = 0
    
    def clear(self):
        self.sample_memory_index *= 0
        self.sample_memory_loc *= 0
        
        self.image_sample_memory *= 0
        self.mask_sample_memory *= 0
        
        self.state_memory = False
        
    def write_memory(self, images, targets, status_update=False):
       bn, dm, fh, fw = images.size()
       sample_memory_ovf = (self.sample_memory_ovf- self.sample_memory_ovf.min()) * self.memory_total_sz + self.sample_memory_loc
       sort_index = np.argsort(sample_memory_ovf)
       for idx_bn in range(bn):
            ltargets = targets[idx_bn]
            label_set = ltargets.get_field("labels").cpu().detach().numpy()
            for idx in range(len(sort_index)):
               label_v = sort_index[idx]
               if (label_v+1) in label_set and ltargets.bbox.size(0) > 1:
                    sample_memory_loc = int(self.sample_memory_loc[label_v])
                    self.images_sample_memory[label_v][sample_memory_loc] = images[idx_bn].cpu()
                    nltargets = ltargets.to('cpu')
                    self.targets_sample_memory[label_v][sample_memory_loc] = nltargets

                    self.sample_memory_loc[label_v] += 1
                    self.sample_memory_index[label_v] += 1
                    if self.sample_memory_index[label_v] > self.memory_total_sz:
                        self.sample_memory_index[label_v] = self.memory_total_sz
                    
                    if self.sample_memory_loc[label_v] == self.memory_total_sz:
                        self.sample_memory_loc[label_v] = 0
                        self.sample_memory_ovf[label_v] += 1
                        
                    break
           
        
       return
    
    def read_memory(self, device):
        if self.sample_memory_index.min() > 2:
            ims = []
            targets = []
            
            sample_memory_index = int(self.sample_memory_index[self.read_sample_index])
            sample_index = random.sample(range(0,sample_memory_index),1)
            
            for idx in sample_index:
                im = self.images_sample_memory[self.read_sample_index][idx]
                target = self.targets_sample_memory[self.read_sample_index][idx]
                
                ims.append(im)
                targets.append(target)
                
            self.read_sample_index += 1
            self.read_sample_index = self.read_sample_index % self.class_numbers
            
            return ims, targets
        else:
            return None, None
            
        
    def forward(self, images=None, targets=None, method='r', device=None):
        if method == 'r':
             return self.read_memory(device)
        elif method == 'w':
             self.write_memory(images, targets)
             return
        
    



