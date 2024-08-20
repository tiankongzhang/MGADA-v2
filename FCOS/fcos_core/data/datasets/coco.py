# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints
import math
import numpy.random as npr

import cv2
import os
import colorsys
from PIL import Image
import numpy as np

import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from .visualizer import Visualizer
 
min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False
    
def _change_color_brightness(color, brightness_factor):
    """
    Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
    less or more saturation than the original color.

    Args:
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
            0 will correspond to no change, a factor in [-1.0, 0) range will result in
            a darker color and a factor in (0, 1.0] range will result in a lighter color.

    Returns:
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mplc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
    return modified_color


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, is_sample=False
    , domain='source'):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)
        self.is_sample = is_sample

        # filter images without detection annotations
        #self.is_train = is_train
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.ntransforms = transforms
        self.domain = domain
        #print('--00:', self.json_category_id_to_contiguous_id)
        #print('--01:', self.contiguous_category_id_to_json_id)

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        #print(img)
        #scale_mark = self.scales[idx]
        scale_p = npr.randint(0,10) / 10.0
        if scale_p < 0.4:
            scale_p = 1.0
        else:
            scale_p = 0.0
        scale_mark = npr.randint(0,2)*2 - 1
        scale_mark = math.pow(npr.randint(2,6), scale_mark * scale_p)
        #print(not self.is_sample)
        if not self.is_sample:
            scale_p = -1.0

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        
        im_w = img.size[0] * scale_mark
        im_h = img.size[1] * scale_mark
        im_cx = im_w / 2
        im_cy = im_h / 2

        crop_im_ex = img.size[0] / 2 - im_cx
        crop_im_ey = img.size[1] / 2 - im_cy

        rl_num_boxes = 0
        bboxes_s = []
        gt_classes_s = []
        
        if scale_p > 0:
            for idx_b in range(len(boxes)):
                bb = boxes[idx_b] * 1
                clsl = classes[idx_b]

                bb[2] = bb[0] + bb[2]
                bb[3] = bb[1] + bb[3]
                if float(bb[2]) - float(bb[0]) > 0 and float(bb[3]) - float(bb[1]) > 0:
                    #bb = (bb.float() * scale_mark).int()

                    bb[0] = int(bb[0] * scale_mark + crop_im_ex)
                    bb[1] = int(bb[1] * scale_mark + crop_im_ey)
                    bb[2] = int(bb[2] * scale_mark + crop_im_ex)
                    bb[3] = int(bb[3] * scale_mark + crop_im_ey)

                    if float(bb[3]) - float(bb[1]) < 8 or float(bb[2]) - float(bb[0]) < 8 or \
                        float(bb[0]) < 0 or float(bb[1]) < 0 or bb[2] > img.size[0] or bb[3] > img.size[1]:
                        continue
                    rl_num_boxes += 1
                    bboxes_s.append(bb)
                    gt_classes_s.append(clsl)

        if rl_num_boxes > 1:
            center = (img.size[0] // 2, img.size[1] // 2)
            scale_mat = cv2.getRotationMatrix2D(center, 0, scale_mark)
            imgv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            img_mask = imgv * 0 + 255
            padding = np.mean(imgv, axis=(0, 1))

            imgv = cv2.warpAffine(imgv, scale_mat, (img.size[0], img.size[1]),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=padding)
            img_mask = cv2.warpAffine(img_mask, scale_mat, (img.size[0], img.size[1]),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
            img = Image.fromarray(cv2.cvtColor(imgv,cv2.COLOR_BGR2RGB))
            fimg_mask = Image.fromarray(cv2.cvtColor(img_mask,cv2.COLOR_BGR2RGB))
            boxes = bboxes_s
            classes = gt_classes_s
        else:
            imgv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            img_mask = imgv * 0 + 255
            fimg_mask = Image.fromarray(cv2.cvtColor(img_mask,cv2.COLOR_BGR2RGB))
            
        
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        classes = torch.tensor(classes)
        
        if rl_num_boxes > 1:
            target = BoxList(boxes, img.size, mode="xyxy")
        else:
            target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        
        target.add_field("labels", classes)

        # masks = [obj["segmentation"] for obj in anno]
        # masks = SegmentationMask(masks, img.size, mode='poly')
        # target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)
        #print(img)
        metas = {'img_mask':fimg_mask, 'domain':self.domain, 'is_sample':self.is_sample}
        if self.ntransforms is not None:
            img, target, metas = self.ntransforms(img, target, metas)
        

        #return imgt, cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR), idx,
        
        #return img, img_o, idx, metas
        
        return img, target, idx, metas

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
