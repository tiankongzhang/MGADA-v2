import math
import torch
import torch.nn.functional as F
from torch import nn

from .loss import make_featuregen_loss_evaluator
from fcos_core.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from fcos_core.layers import Scale
from fcos_core.modeling.poolers import Pooler
#from cvpods.layers.deform_conv_with_off import DeformConvWithOff, ModulatedDeformConvWithOff

#simple global feature
class getTwoStageFeature(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(getTwoStageFeature, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.zero_prev = nn.Sequential(
             nn.AvgPool2d((1,1), stride=1),
             nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
             #nn.ReLU(),
        )

        
        self.one_prev = nn.Sequential(
            nn.AvgPool2d((3,3), stride=2, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.ReLU()
            )
        
        
        self.conv_anchors = [
            [3,5], [5,3], [3,3]
        ]

        self.padding_anchors = [
            [1,2], [2,1], [1,1]
        ]

        for idx in range(len(self.conv_anchors)):
            conv_anchor = self.conv_anchors[idx]
            padding_anchor = self.padding_anchors[idx]
            self.add_module('glo'+str(idx+2),
                    nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=(conv_anchor[0], conv_anchor[1]), \
                        stride=1, padding=(padding_anchor[0], padding_anchor[1]), bias=False),
                        #nn.ReLU()
                    )
                )
            
        for idx in range(len(self.conv_anchors)):
            conv_anchor = self.conv_anchors[idx]
            padding_anchor = self.padding_anchors[idx]
            self.add_module('glt'+str(idx+2),
                    nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=(conv_anchor[0], conv_anchor[1]), \
                        stride=1, padding=(padding_anchor[0], padding_anchor[1]), bias=False),
                        #nn.ReLU()
                    )
                )
        
        
    def forward(self, xs):
        #zero stage
        lznx = self.zero_prev(xs)
        lonx = self.one_prev(lznx)
        
        
        onx = []
        for idx in range(len(self.conv_anchors)):
            gl = getattr(self, 'glo'+str(idx+2))
            onx.append(gl(lznx))
        
        tnx = []
        for idx in range(len(self.conv_anchors)):
            gl = getattr(self, 'glo'+str(idx+2))
            gnx = gl(lonx)
            gnx = F.interpolate(gnx, size=[xs.size(-2), xs.size(-1)], mode="bilinear")
            tnx.append(gnx)
            
        return {'one':onx, 'two':tnx}



class getGlobal(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(getGlobal, self).__init__()
        # TODO: Implement the sigmoid version first.
        levels = len(cfg.MODEL.GENFEATURE.FPN_STRIDES)
        self.msf = getTwoStageFeature(cfg, in_channels)
        
    def forward(self, x):
        #two stage
        nx = []
        for idx in range(len(x)):
            xs = self.msf(x[idx])
            nx.append(xs)
        return nx


class FeatureGenModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(FeatureGenModule, self).__init__()
        self.fpn_strides = cfg.MODEL.GENFEATURE.FPN_STRIDES
        self.local_global_merge = cfg.MODEL.GENFEATURE.LOCAL_GLOBAL_MERGE
        self.glFeature = getGlobal(cfg, in_channels)

        if cfg.MODEL.GENFEATURE.LOCAL_GLOBAL_MERGE:
            self.upsample_conv = nn.Sequential(
                nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                #nn.ReLU(True)
            )
            
            if "R-101" in cfg.MODEL.BACKBONE.CONV_BODY:
                #Resnet-101
                dims = [512, 1024, 2048, 256, 256]
            else:
                #Vgg-16
                dims = [256, 512, 512, 256, 256]
                
            for idx in range(len(self.fpn_strides)):
                self.add_module('down_sample'+str(idx+2),
                    nn.Sequential(nn.Conv2d(dims[idx], in_channels, kernel_size=1, stride=1, bias=False),
                    #nn.ReLU()
                    ))


    def forward(self, pre_features, features, box_regression, im_sz, targets=None, return_maps=False):
        npre_features = []
        for idx in range(len(pre_features)):
            object_local_convs = getattr(self, 'down_sample'+str(idx+2))
            npre_features.append(object_local_convs(pre_features[idx]))
            
        hfeatures = self.glFeature(npre_features)
        locations = self.compute_locations(features)
        nfeatures = []
        for level in range(len(features)):
             features_dict = {}
             for key, value in hfeatures[level].items():
                 feature_list = []
                 for idx in range(len(value)):
                     lvalue = self.upsample_conv(value[idx])
                     feature_list.append(lvalue)
                 features_dict.update({key:feature_list})
             
             nfeatures.append(features_dict)
            
        #weights
        bbs_iou_weights_list = []
        for level in range(len(features)):
            conv_anchors = [
                [1,2], [2,1], [1,1],
                [3,5], [5,3], [3,3]
            ]
            
            
            ##one kernel area
            lproposals_im = box_regression[level] * 1.0 / self.fpn_strides[level]
            lproposals_im[:, 2]  = (lproposals_im[:, 2] + lproposals_im[:, 0]) / 2
            lproposals_im[:, 3]  = (lproposals_im[:, 3] + lproposals_im[:, 1]) / 2
            lproposals_im[:,0] = lproposals_im[:, 2]
            lproposals_im[:,1] = lproposals_im[:, 3]
            lproposals_im[lproposals_im <= 0] = 1.0
            
            bbs_iou_weights = []
            for ii in range(len(conv_anchors)):
                conv_bbs = self.generate_conv_box(lproposals_im, conv_anchors[ii], self.fpn_strides[level])

                boxes_iou = self.IoU(lproposals_im, conv_bbs)
                bbs_iou_weights.append(boxes_iou)

            bbs_iou_weights = torch.stack(bbs_iou_weights, dim=1)
            bbs_iou_weights = bbs_iou_weights - bbs_iou_weights.max(dim=1,keepdim=True)[0]/( bbs_iou_weights.max(dim=1,keepdim=True)[0] + 1e-5)
            bbs_iou_weights_list.append(F.softmax(1.0 * bbs_iou_weights, dim=1))

        #box
        boxes_list = []
        for level in range(len(features)):
            ft_ll = box_regression[level]
            im_num = ft_ll.size(0)

            boxes_im = []
            for im in range(im_num):
                location_l = locations[level].view(-1, 2)
                box_regression_l = box_regression[level][im].permute(1, 2, 0).view(-1, 4)
                nc, nh, nw = features[level][im].size()

                x1 = ((location_l[:, 0] - box_regression_l[:, 0]) / self.fpn_strides[level]).clamp(min=0, max=nw - 1)
                y1 = ((location_l[:, 1] - box_regression_l[:, 1]) / self.fpn_strides[level]).clamp(min=0, max=nh - 1)
                x2 = ((location_l[:, 0] + box_regression_l[:, 2]) / self.fpn_strides[level]).clamp(min=0, max=nw - 1)
                y2 = ((location_l[:, 1] + box_regression_l[:, 3]) / self.fpn_strides[level]).clamp(min=0, max=nh - 1)

                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                boxes = boxes.view(nh, nw, 4)
                boxes_im.append(boxes)
            boxes_list.append(boxes_im)

        #feature alginment
        align_features_list = []
        for level in range(len(features)):
            glfeatures = nfeatures[level]
            im_num, nc, nh, nw = features[level].size()

            one_gl_xs0_list = []
            one_gl_xs1_list = []
            one_gl_xs2_list = []
            one_gl_xs3_list = []

            two_gl_xs0_list = []
            two_gl_xs1_list = []
            two_gl_xs2_list = []
            two_gl_xs3_list = []

            feature_multis = []

            one_f = []
            two_f = []
            for im in range(im_num):
                boxes = boxes_list[level][im] * 1
                boxes = boxes.view(-1, 4)
                cx = ((boxes[:,0] + boxes[:,2]) / 2).clamp(min=0, max=nw - 1)
                cy = ((boxes[:,1] + boxes[:,3]) / 2).clamp(min=0, max=nh - 1)
                c_loc = (cy * nw + cx).long()
                #one
                one_gl_xs0 = glfeatures['one'][0][im].view(-1, nh*nw)
                one_gl_xs0 = torch.index_select(one_gl_xs0, 1, c_loc).view(nc, nh, nw)

                one_gl_xs1 = glfeatures['one'][1][im].view(-1, nh*nw)
                one_gl_xs1 = torch.index_select(one_gl_xs1, 1, c_loc).view(nc, nh, nw)

                one_gl_xs2 = glfeatures['one'][2][im].view(-1, nh*nw)
                one_gl_xs2 = torch.index_select(one_gl_xs2, 1, c_loc).view(nc, nh, nw)


                #one
                two_gl_xs0 = glfeatures['two'][0][im].view(-1, nh*nw)
                two_gl_xs0 = torch.index_select(two_gl_xs0, 1, c_loc).view(nc, nh, nw)

                two_gl_xs1 = glfeatures['two'][1][im].view(-1, nh*nw)
                two_gl_xs1 = torch.index_select(two_gl_xs1, 1, c_loc).view(nc, nh, nw)

                two_gl_xs2 = glfeatures['two'][2][im].view(-1, nh*nw)
                two_gl_xs2 = torch.index_select(two_gl_xs2, 1, c_loc).view(nc, nh, nw)

                feature_multi = torch.stack((one_gl_xs0, one_gl_xs1, one_gl_xs2, two_gl_xs0, two_gl_xs1, two_gl_xs2), dim=0)
                feature_multis.append(feature_multi)


            feature_multis = torch.stack(feature_multis, dim=0)
            align_features_list.append(feature_multis)


        #feature selection
        align_gl_nfeatures_list = []
        for level in range(len(features)):
            bbs_iou_weights = bbs_iou_weights_list[level].detach()
            feature_multis = align_features_list[level]
            B, N, C, H, W = feature_multis.size()
            align_gl_feature = feature_multis.mul(bbs_iou_weights.view(B, N, 1, H, W)).sum(dim=1)

            if self.local_global_merge:
                local_ft = self.upsample_conv(features[level])
                align_gl_features = 1.0 * align_gl_feature + local_ft
                align_gl_nfeatures_list.append(align_gl_features)
            else:
                align_gl_nfeatures_list.append(align_gl_feature)

        return align_gl_nfeatures_list

    def generate_conv_box(self, box_regression, bbs, strides=8):
        conv_bbs = torch.zeros_like(box_regression)
        
        conv_bbs[:, 0] += bbs[1]
        conv_bbs[:, 2] += bbs[1]
        conv_bbs[:, 1] += bbs[0]
        conv_bbs[:, 3] += bbs[0]
        
        return conv_bbs


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

    def IoU(self, pred, target, weight=None):
        pred_left = (pred[:, 0] + pred[:, 2]) / 2
        pred_top = (pred[:, 1] + pred[:, 3]) / 2
        pred_right = pred_left
        pred_bottom = pred_top

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        area_union = area_union.clamp(min=0.0)
        ac_uion = ac_uion.clamp(min=1e-5)

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        return ious

def build_featuregen(cfg, in_channels):
    return FeatureGenModule(cfg, in_channels)
