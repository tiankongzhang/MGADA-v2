# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as FF
#from torchvision.transforms.functional

import cv2
from .aug2_utils import *

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, metas):
        for t in self.transforms:
            image, target, metas = t(image, target, metas)
        return image, target, metas

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target, metas):
        size = self.get_size(image.size)
        image = FF.resize(image, size)
        target = target.resize(image.size)
        return image, target, metas


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target, metas):
        #print('---1')
        if random.random() < self.prob:
            image = FF.hflip(image)
            target = target.transpose(0)
        return image, target, metas


class ToTensor(object):
    def __call__(self, image, target, metas):
        RESIZE = image.size[0]
        RCROP = [image.size[1], image.size[0]]
        
        if metas['domain'] == 'source':
            SCALE_SZ0 = [1.0, 2.0, 3.5]
            #SCALE_SZ0 = [1.0, 1.02]
            aug_scale = ADCompose([
                        ADAdjustGamma(0.2),
                        ADAdjustSaturation(0.2),
                        ADAdjustHue(0.2),
                        ADAdjustContrast(0.2),
                        ADRandomLSized(RESIZE, SCALE_SZ0),
                        ADCenterCrop(RCROP),
                        #ADRandomHorizontallyFlip(0.15)]
                        ])
        else:
            SCALE_SZ0 = [1.0, 2.0]
            aug_scale = ADCompose([
                        ADAdjustGamma(0.2),
                        ADAdjustSaturation(0.2),
                        ADAdjustHue(0.2),
                        ADAdjustContrast(0.2),
                        ADRandomLSized(RESIZE, SCALE_SZ0),
                        ADCenterCrop(RCROP),
                        #ADRandomHorizontallyFlip(0.15)]
                        ])
        
        aug_transfer = ADCompose([
                    ADAdjustGamma(0.2),
                    ADAdjustSaturation(0.2),
                    ADAdjustHue(0.2),
                    ADAdjustContrast(0.2),
                    ADRandomLSized(RESIZE, SCALE_SZ1),
                    ADCenterCrop(RCROP)
                    ])
        
        
        ##scale samples
        sc_img, sc_tran_dict, sc_mask= aug_scale(image, metas['img_mask'])
        metas['sc_tran_dict'] = sc_tran_dict
        sc_img = sc_img
        tr_img, tr_tran_dict, tr_mask = aug_transfer(image, metas['img_mask'])
        metas['tr_tran_dict'] = tr_tran_dict
        tr_img = tr_img#
        
        if metas['is_sample']:
            nimage, _, _ = aug_transfer(image)
        else:
            nimage = image
        
        metas['sc_img'] = FF.to_tensor(sc_img)
        metas['tr_img'] = FF.to_tensor(tr_img)
        metas['img_mask_sc'] = FF.to_tensor(sc_mask)
        metas['img_mask_tr'] = FF.to_tensor(tr_mask)
        
        return FF.to_tensor(nimage), target, metas


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target, metas):
        
        
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
            sc_img = metas['sc_img'][[2, 1, 0]] * 255
            tr_img = metas['tr_img'][[2, 1, 0]] * 255
        else:
            image = image
            sc_img = metas['sc_img']
            tr_img = metas['tr_img']
            
        image = FF.normalize(image, mean=self.mean, std=self.std)
        sc_img = FF.normalize(sc_img, mean=self.mean, std=self.std)
        tr_img = FF.normalize(tr_img, mean=self.mean, std=self.std)
        
        metas['sc_img'] = sc_img
        metas['tr_img'] = tr_img
        
        return image, target, metas
