import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor, make_fcos_2_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)
        
        cfg_train_test = {'INFERENCE_TH': 0.42, 'PRE_NMS_TOP_N': 1000, 'NMS_TH': 0.5, 'DETECTIONS_PER_IMG': 20, 'NUM_CLASSES':cfg.MODEL.FCOS.NUM_CLASSES, 'THRESH_MODE':0}
        box_selector_train_test = make_fcos_2_postprocessor(cfg_train_test)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.box_selector_train_test = box_selector_train_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None, return_maps=False, box_regression_coarse=None, is_meanT=False):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            feat_weights = []
            boxes = None
            if is_meanT:
                #norm
                boxes = self.box_selector_train_test(
                    locations, box_cls, box_regression,
                    centerness, images.image_sizes
                )
                
                if targets is None:
                    number_bbs = 0
                    for idx in range(len(boxes)):
                        lbbox = boxes[idx].bbox
                        number_bbs += lbbox.size(0)
                        
                    if number_bbs <= 0:
                        boxes = self.box_selector_train_test(
                            locations, box_cls, box_regression,
                            centerness, images.image_sizes,
                            pre_nms_thresh = 0.3,
                            thresh_mode=1, pre_nms_top_n=3,
                        )
                
                if targets is None:
                    with torch.no_grad():
                        for idx in range(len(features)):
                            labels, _ = self.compute_targets_for_locations(locations[idx], boxes, features[idx])
                            feat_weights.append(labels.detach())
                else:
                    with torch.no_grad():
                        for idx in range(len(features)):
                            labels, _ = self.compute_targets_for_locations(locations[idx], targets, features[idx])
                            feat_weights.append(labels.detach())
            
            _, losses, outputs, weights = self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, targets, return_maps,
                box_regression_coarse
            
            )
            
            return boxes, losses, outputs, feat_weights
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes,
                targets=targets, box_regression_coarse=box_regression_coarse
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets, return_maps=False, box_regression_coarse=None):
        score_maps = {
            "box_cls": box_cls,
            "box_regression": box_regression,
            "centerness": centerness
        }
        losses = {}
        if targets is not None:
            loss_box_cls, loss_box_reg, loss_centerness, weights = self.loss_evaluator(
                locations, box_cls, box_regression, centerness, targets, box_regression_coarse
            )

            losses = {
                "loss_cls": 1.0 * loss_box_cls,
                "loss_reg": 1.0 * loss_box_reg,
                "loss_centerness": 1.0 * loss_centerness
            }
            
        else:
            loss_cls = 0.0 * sum(0.0 * torch.sum(x) for x in box_cls)
            loss_reg = 0.0 * sum(0.0 * torch.sum(x) for x in box_regression)
            loss_centerness = 0.0 * sum(0.0 * torch.sum(x) for x in centerness)
            losses = {
                "zero": loss_cls + loss_reg + loss_centerness
            }
            
            weights = []
            for idx in range(len(box_cls)):
                lbb_cls = box_cls[idx].sigmoid()
                lbb_ctr = centerness[idx].sigmoid()
                mpred = lbb_cls.mul(lbb_ctr)
                mpred_v, mpred_i = mpred.max(dim=1)
                
                weight_0 = (mpred_v > 0.01).long()
                weight = mpred_i.mul(weight_0) + weight_0
                weights.append(weight)
        
                
        if return_maps:
            outputs_maps = []
            for idx in range(len(box_cls)):
                outputs_maps.append([box_cls[idx].sigmoid(), box_regression[idx], centerness[idx].sigmoid()])
            
            outputs = {'outputs_maps':outputs_maps}
            
            
            return None, losses, score_maps, weights
        else:
            return None, losses, None, None

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes, targets=None, box_regression_coarse=None):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        losses = {}
        if targets is not None:
            loss_box_cls, loss_box_reg, loss_centerness, weights = self.loss_evaluator(
                locations, box_cls, box_regression, centerness, targets, box_regression_coarse
            )
            losses = {
                "loss_cls": loss_box_cls,
                "loss_reg": loss_box_reg,
                "loss_centerness": loss_centerness
            }
            
            
        return boxes, losses, None, None
    
    def compute_targets_for_locations(self, locations, targets, feats):
        bn, lc, lh, lw = feats.size()
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        for im_i in range(bn):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field('labels')
            if bboxes.size(0) == 0:
                labels_per_im = torch.zeros_like(locations[:,0])
                
                reg_targets_per_im = torch.zeros_like(torch.cat([locations, locations], dim=1))
                labels_per_im = labels_per_im.long()
                
            else:
                l = xs[:, None] - bboxes[:, 0][None]
                t = ys[:, None] - bboxes[:, 1][None]
                r = bboxes[:, 2][None] - xs[:, None]
                b = bboxes[:, 3][None] - ys[:, None]
                reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
                max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
                locations_to_gt_area = torch.ones_like(max_reg_targets_per_im) 
                locations_to_gt_area[is_in_boxes == 0] = 0

                # if there are still more than one objects for a location,
                # we choose the one with minimal area
                locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.max(dim=1)

                reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
                labels_per_im = labels_per_im[locations_to_gt_inds]
                labels_per_im[locations_to_min_area == 0] = 0
                
            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
        
        labels = torch.stack(labels, dim=0)
        labels = labels.view(bn, lh, lw)
        
        reg_targets = torch.stack(reg_targets, dim=0)
        reg_targets = reg_targets.view(bn, lh, lw, 4)
        
        return labels, reg_targets

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    
def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
