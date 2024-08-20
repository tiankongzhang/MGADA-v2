import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import random
#from kmeans_pytorch import kmeans
from .SampleMemory import SamplerMemory
from .SampleAugmentation import SamplerAugmentation

class MemoryManagement(nn.Module):
    def __init__(self, num_classes):
        super(MemoryManagement, self).__init__()
        self.class_numbers = num_classes
        self.source_memory = SamplerMemory(self.class_numbers)
        self.source_aug = SamplerAugmentation(self.class_numbers, is_translate=True)
        self.target_memory = SamplerMemory(self.class_numbers)
        self.target_aug = SamplerAugmentation(self.class_numbers)
        
    def forward(self, images=None, targets=None, method='r', device=None, domain='s'):
        if method == 'r':
             if domain == 's':
                 limages, ltargets = self.source_memory(device)
                 return {'images': limages, 'targets': ltargets}
             elif domain == 't':
                 limages, ltargets = self.target_memory(device)
                 return {'images': limages, 'targets': ltargets}
        elif method == 'w':
             if domain == 's':
                 images, targets = self.source_aug(images, targets)
                 self.source_memory(images, targets, method='w')
             elif domain == 't':
                 images, targets = self.source_aug(images, targets)
                 self.target_memory(images, targets, method='w')
        
    



