import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
import random
from torchvision import transforms

class SamplerAugmentation(nn.Module):
    def __init__(self, num_classes, im_szs, is_translate=False):
        super(SamplerAugmentation, self).__init__()
        
    
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        
        feat = feat.permute(1,0,2,3).contiguous()
        feat_var = feat.view(C, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(1, C, 1, 1)
        feat_mean = feat.view(C, -1).mean(dim=1).view(1, C, 1, 1)
        return feat_mean, feat_std

    #AadIN
    def adaptive_instance_normalization(self, content_feat, g_value=None, b_value=None):
        size = content_feat.size()
        g_value = g_value.view(size[0],1,1,1)
        b_value = b_value.view(size[0],1,1,1)
        
        content_mean, content_std = self.calc_mean_std(content_feat)
        gamma_ = content_std.mul(content_std.clone().detach().uniform_(0.8, 0.9))
        beta_ = content_mean.mul(content_mean.clone().detach().uniform_(0.4, 0.5))

        beta_ = beta_.mul(content_mean.div(content_std))
        beta = beta_.mul(b_value)
        gamma = gamma_.mul(g_value)

        normalized_feat = (content_feat - content_mean) / content_std
        normalized_feat = normalized_feat * gamma + beta
        
        return normalized_feat
    
        
    def forward(self, images, target):
        B, C, H, W = images.size()
        
        #cross
        g_value_ = torch.randn((B)).to(images.device)
        b_value_ = torch.randn((B)).to(images.device)
        
        b_value_ = b_value_.uniform_(1.0, 1.001)
        g_value_ = g_value_.uniform_(1.0, 1.001)
        images = self.adaptive_instance_normalization(images, g_value=g_value_, b_value=b_value_)
        
        return images, target



