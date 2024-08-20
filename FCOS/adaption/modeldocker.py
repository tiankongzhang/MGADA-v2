import numpy as np
import random
import os
from functools import partial
import cv2
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from terminaltables import AsciiTable
from fcos_core.utils.model_zoo import cache_url
from fcos_core.utils.imports import import_file
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.model_serialization import load_state_dict
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.box_generator.level_box import build_boxgen
from fcos_core.modeling.feature_generator.level_feature import build_featuregen
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.structures.image_list import to_image_list
from fcos_core.structures.bounding_box import BoxList
from fcos_core.modeling.discriminator import FCOSDiscriminator, FCOSDiscriminator_CA, FCOSDiscriminator_CC
from fcos_core.utils.c2_model_loading import load_c2_format


class DockerModel(nn.Module):
    def __init__(self, adap_cfg):
        super(DockerModel, self).__init__()
        self.cfg = adap_cfg
        self.class_num = adap_cfg.MODEL.FCOS.NUM_CLASSES
        
        self.backbone_type = adap_cfg.MODEL.BACKBONE.CONV_BODY
        self.use_teacher = adap_cfg.MODEL.MODE.USE_TEACHER
        self.use_gate = adap_cfg.MODEL.MODE.USE_GATE
        self.gate_fusion = adap_cfg.MODEL.MODE.GATE_FUSION
        
        self.teacher_backbone = build_backbone(adap_cfg)
        if self.use_gate:
            #gen box
            self.teacher_genbox = build_boxgen(adap_cfg, self.teacher_backbone.out_channels)
            #gen new feature
            self.teacher_genfeature = build_featuregen(adap_cfg, self.teacher_backbone.out_channels)
        self.teacher_fcos = build_rpn(adap_cfg, self.teacher_backbone.out_channels)
        self.init_teacher_weights(pretained=adap_cfg.MODEL.PRETAINED)
        
        
        if adap_cfg.MODEL.MODE.USE_STUDENT:
             self.student_backbone = build_backbone(adap_cfg)
             if self.use_gate:
                 #gen box
                 self.student_genbox = build_boxgen(adap_cfg, self.student_backbone.out_channels)
                 #gen new feature
                 self.student_genfeature = build_featuregen(adap_cfg, self.student_backbone.out_channels)
             self.student_fcos = build_rpn(adap_cfg, self.student_backbone.out_channels)
             self.init_student_weights(pretained=adap_cfg.MODEL.PRETAINED)
        
        
        ##DIS
        self.used_feature_layers = adap_cfg.MODEL.MODE.USED_FEATURE_LAYERS
        if adap_cfg.MODEL.DETECT.USE_DIS_GLOBAL:
            for idx in range(len(self.used_feature_layers)):
                self.add_module('ddis'+str(idx+3),
                        FCOSDiscriminator(
                            num_convs=adap_cfg.MODEL.DETECT.DIS_NUM_CONVS,
                            grad_reverse_lambda=adap_cfg.MODEL.DETECT.GRL_WEIGHT_PL,
                            grl_applied_domain=adap_cfg.MODEL.DETECT.GRL_APPLIED_DOMAIN
                        )
                    )

        if adap_cfg.MODEL.ADV.USE_DIS_GLOBAL:
            for idx in range(len(self.used_feature_layers)):
                self.add_module('fdis'+str(idx+3),
                        FCOSDiscriminator(
                            num_convs=adap_cfg.MODEL.ADV.DIS_NUM_CONVS,
                            grad_reverse_lambda=adap_cfg.MODEL.ADV.GRL_WEIGHT_PL,
                            grl_applied_domain=adap_cfg.MODEL.ADV.GRL_APPLIED_DOMAIN
                        )
                    )
        
        if adap_cfg.MODEL.CM.USE_CM_GLOBAL:
            for idx in range(len(self.used_feature_layers)):
                self.add_module('dis_cc'+str(idx+3),
                        FCOSDiscriminator_CC(
                            adap_cfg,
                            num_convs=adap_cfg.MODEL.CM.DIS_NUM_CONVS,
                            grad_reverse_lambda=adap_cfg.MODEL.CM.GRL_WEIGHT_PL,
                            grl_applied_domain=adap_cfg.MODEL.CM.GRL_APPLIED_DOMAIN
                        )
                    )
        
        self.USE_DIS_DETECT_GL = adap_cfg.MODEL.DETECT.USE_DIS_GLOBAL
        self.USE_DIS_GLOBAL = adap_cfg.MODEL.ADV.USE_DIS_GLOBAL
        self.USE_CC_GLOBAL = adap_cfg.MODEL.CM.USE_CM_GLOBAL
        
        self.dt_dis_lambda = adap_cfg.MODEL.DETECT.DT_DIS_LAMBDA
        self.ga_dis_lambda = adap_cfg.MODEL.ADV.GA_DIS_LAMBDA
        self.cm_dis_lambda = adap_cfg.MODEL.CM.GL_CM_LAMBDA
             
        self.update_teacher_interum = adap_cfg.MODEL.MODE.UPDATE_TEACHER_ITERUM
        self.NET_MOMENTUM = adap_cfg.MODEL.MODE.NET_MOMENTUM
    
    
    def init_teacher_weights(self, pretained=None):
        if pretained is not None and pretained != '':
              pickle.load = partial(pickle.load, encoding='latin-1')
              pickle.Unpickler = partial(pickle.Unpickler, encoding='latin-1')
              if pretained.startswith("http"):
                  cached_f = cache_url(pretained)
                  pretained = cached_f
              
              if pretained.startswith("catalog://"):
                  PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
                  paths_catalog = import_file(
                      "maskrcnn_benchmark.config.paths_catalog", PATHS_CATALOG, True
                  )
                  catalog_f = paths_catalog.ModelCatalog.get(pretained[len("catalog://") :])
                  pretained = catalog_f
              
              if 'R-101' in pretained:
                  checkpoint = load_c2_format(self.cfg, pretained)
              else:
                  checkpoint = torch.load(pretained, map_location=torch.device("cpu"))
                  
              if "model_backbone" in checkpoint.keys():
                  load_state_dict(self.teacher_backbone, checkpoint.pop("model_backbone"))
                  load_state_dict(self.teacher_fcos, checkpoint.pop("model_fcos"))
              elif 'model' in checkpoint.keys():
                  load_state_dict(self.teacher_backbone, checkpoint.pop("model"))
              else:
                  load_state_dict(self.teacher_backbone, checkpoint)
    
    
    def init_student_weights(self, pretained=None):
        if pretained is not None and pretained != '':
              if pretained.startswith("http"):
                  cached_f = cache_url(pretained)
                  pretained = cached_f
              
              if pretained.startswith("catalog://"):
                  PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
                  paths_catalog = import_file(
                      "maskrcnn_benchmark.config.paths_catalog", PATHS_CATALOG, True
                  )
                  catalog_f = paths_catalog.ModelCatalog.get(pretained[len("catalog://") :])
                  pretained = catalog_f
              
              if 'R-101' in pretained:
                  checkpoint = load_c2_format(self.cfg, pretained)
              else:
                  checkpoint = torch.load(pretained, map_location=torch.device("cpu"))
              
              
              if "model_backbone" in checkpoint.keys():
                  load_state_dict(self.student_backbone, checkpoint.pop("model_backbone"))
                  load_state_dict(self.student_fcos, checkpoint.pop("model_fcos"))
              elif 'model' in checkpoint.keys():
                  load_state_dict(self.student_backbone, checkpoint.pop("model"))
              else:
                  load_state_dict(self.student_backbone, checkpoint)
    
    def use_teacher_network(self, images, targets=None, return_maps=False, is_meanT=False):
         map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
         feature_layers = map_layer_to_index.keys()
         model_backbone = self.teacher_backbone
         if self.use_gate:
             model_genbox = self.teacher_genbox
             model_genfeature = self.teacher_genfeature
         model_fcos = self.teacher_fcos
         
         images = to_image_list(images)
         pre_features = model_backbone.body(images.tensors)
         features = model_backbone.fpn(pre_features)

         if self.use_gate:
             if "R-101" in self.backbone_type:
                 npre_features = [pre_features[1], pre_features[2], pre_features[3], features[-2], features[-1]]
             else:
                 npre_features = [pre_features[2], pre_features[3], pre_features[4], features[-2], features[-1]]
             f_dt = {
                 layer: features[map_layer_to_index[layer]]
                 for layer in feature_layers
             }
             
             losses = {}
             if model_fcos.training and targets is None:
                 # train G on target domain
                 _, detector_loss, detector_maps = model_genbox(images, features, targets=None, return_maps=return_maps)
                 features_gl = model_genfeature(npre_features, features, detector_maps['box_regression'], images.tensors.size(), targets=None, return_maps=return_maps)
                 
                 proposals, proposal_losses, score_maps, weights = model_fcos(
                     images, features_gl, targets=None, return_maps=return_maps, box_regression_coarse=detector_maps['box_regression'], is_meanT=is_meanT)
                 
             else:
                 # train G on source domain / inference
                 _, detector_loss, detector_maps = model_genbox(images, features, targets=targets, return_maps=return_maps)
                 features_gl = model_genfeature(npre_features, features, detector_maps['box_regression'], images.tensors.size(), targets=targets, return_maps=return_maps)
                 
                 proposals, proposal_losses, score_maps, weights = model_fcos(
                     images, features_gl, targets=targets, return_maps=return_maps, box_regression_coarse=detector_maps['box_regression'], is_meanT=is_meanT)
             
             #global feature
             f_gl = {
                 layer: features_gl[map_layer_to_index[layer]]
                 for layer in feature_layers
             }
         else:
             #local feature
             f_dt = None
             losses = {}
             if model_fcos.training and targets is None:
                 # train G on target domain
                 proposals, proposal_losses, score_maps, weights = model_fcos(
                     images, features, targets=None, return_maps=return_maps, box_regression_coarse=None, is_meanT=is_meanT)
                 
             else:
                 # train G on source domain / inference
                 proposals, proposal_losses, score_maps, weights = model_fcos(
                     images, features, targets=targets, return_maps=return_maps, box_regression_coarse=None, is_meanT=is_meanT)
             
             #global feature
             f_gl = {
                 layer: features[map_layer_to_index[layer]]
                 for layer in feature_layers
             }
         
         losses = {}
         if model_fcos.training:
             m = {
                 layer: {
                     map_type:
                     score_maps[map_type][map_layer_to_index[layer]]
                     for map_type in score_maps
                 }
                 for layer in feature_layers
             }
             
             weight_set = {}
             if is_meanT:
                 weight_set = {
                     layer: weights[map_layer_to_index[layer]]
                     for layer in feature_layers
                 }
         else:
             m = {}
             weight_set = {}
         losses.update(proposal_losses)
         if self.use_gate and self.gate_fusion==0:
            losses.update(detector_loss)
         
         return losses, f_dt, f_gl, m, weight_set, proposals
    
    def use_student_network(self, images, targets=None, return_maps=False, is_meanT=False, input_type=0):
        map_layer_to_index = {"P3": 0, "P4": 1, "P5": 2, "P6": 3, "P7": 4}
        feature_layers = map_layer_to_index.keys()
        model_backbone = self.student_backbone
        if self.use_gate:
            model_genbox = self.student_genbox
            model_genfeature = self.student_genfeature
        model_fcos = self.student_fcos
        
        images = to_image_list(images)
        pre_features = model_backbone.body(images.tensors)
        features = model_backbone.fpn(pre_features)
        
        if self.use_gate:
            if "R-101" in self.backbone_type:
                npre_features = [pre_features[1], pre_features[2], pre_features[3], features[-2], features[-1]]
            else:
                npre_features = [pre_features[2], pre_features[3], pre_features[4], features[-2], features[-1]]
            #local feature
            f_dt = {
                layer: features[map_layer_to_index[layer]]
                for layer in feature_layers
            }
            
            losses = {}
            if model_fcos.training and targets is None:
                # train G on target domain
                _, detector_loss, detector_maps = model_genbox(images, features, targets=None, return_maps=return_maps)
                features_gl = model_genfeature(npre_features, features, detector_maps['box_regression'], images.tensors.size(), targets=None, return_maps=return_maps)
                
                proposals, proposal_losses, score_maps, weights = model_fcos(
                    images, features_gl, targets=None, return_maps=return_maps, box_regression_coarse=detector_maps['box_regression'], is_meanT=is_meanT)
            else:
                # train G on source domain / inference
                _, detector_loss, detector_maps = model_genbox(images, features, targets=targets, return_maps=return_maps)
                features_gl = model_genfeature(npre_features, features, detector_maps['box_regression'], images.tensors.size(), targets=targets, return_maps=return_maps)
                
                proposals, proposal_losses, score_maps, weights = model_fcos(
                    images, features_gl, targets=targets, return_maps=return_maps, box_regression_coarse=detector_maps['box_regression'], is_meanT=is_meanT)
            
            #global feature
            f_gl = {
                layer: features_gl[map_layer_to_index[layer]]
                for layer in feature_layers
            }
        else:
            #local feature
            f_dt = None
            
            losses = {}
            if model_fcos.training and targets is None:
                # train G on target domain
                proposals, proposal_losses, score_maps, weights = model_fcos(
                    images, features, targets=None, return_maps=return_maps, box_regression_coarse=None, is_meanT=is_meanT)
                
            else:
                # train G on source domain / inference
                proposals, proposal_losses, score_maps, weights = model_fcos(
                    images, features, targets=targets, return_maps=return_maps, box_regression_coarse=None, is_meanT=is_meanT)
            
            #global feature
            f_gl = {
                layer: features[map_layer_to_index[layer]]
                for layer in feature_layers
            }
        
        losses = {}
        if model_fcos.training:
            m = {
                layer: {
                    map_type:
                    score_maps[map_type][map_layer_to_index[layer]]
                    for map_type in score_maps
                }
                for layer in feature_layers
            }
            
            weight_set = {}
            if is_meanT:
                weight_set = {
                    layer: weights[map_layer_to_index[layer]]
                    for layer in feature_layers
                }
        else:
            m = {}
            weight_set = {}
        losses.update(proposal_losses)
        if self.use_gate and self.gate_fusion==0:
           losses.update(detector_loss)
        
        return losses, f_dt, f_gl, m, weight_set, proposals
    

    def forward(self, tc_images, targets_=None, domain='source', st1_images=None, st1_trans=None, st2_images=None, mode=0, features_lc_s=None, features_gl_s=None, gt_weights = None, valid=False, val_net=0, ims_mask_st1=None, ims_mask_st2=None):
        #output dict
        outputs = dict()
        loss_dict = dict()
        if self.training:
            if self.cfg.MODEL.MODE.TRAIN_PROCESS == 'S0':
                 return self.use_teacher_network(tc_images, targets_, return_maps=True)
            else:
                
                if mode == 0:
                    if targets_ is None:
                        if self.cfg.MODEL.MODE.TRAIN_PROCESS == 'SL':
                            return self.use_student_network(tc_images, targets_, return_maps=True, is_meanT=True)
                        else:
                            ims_mask_st1 = to_image_list(ims_mask_st1).tensors
                            ims_mask_st2 = to_image_list(ims_mask_st2).tensors
                            with torch.no_grad():
                                if self.use_teacher:
                                    loss_dict, features_lc_s, features_gl_s, score_maps_s, weights, proposals =  self.use_teacher_network(tc_images, targets_, return_maps=True, is_meanT=True)
                                else:
                                    loss_dict, features_lc_s, features_gl_s, score_maps_s, weights, proposals =  self.use_student_network(tc_images, targets_, return_maps=True, is_meanT=True)
                                    
                                targets_ = self.generate_target(proposals)
                                stsc_images, st_targets_, st_ims_mask = self.tranformer_target(st1_images, targets_, st1_trans, ims_mask_st1, st2_images, ims_mask_st2,vis=True)
                            return stsc_images, st_targets_, st_ims_mask
                            
                    elif domain == 'source':
                        ims_mask_st1 = to_image_list(ims_mask_st1).tensors
                        ims_mask_st2 = to_image_list(ims_mask_st2).tensors
                        stsc_images, st_targets_, st_ims_mask = self.tranformer_target(st1_images, targets_, st1_trans, ims_mask_st1, st2_images, ims_mask_st2)
                        losses, f_dt, f_gl, m, weight_set, proposals = self.use_student_network(stsc_images, st_targets_, return_maps=True, is_meanT=True)
                        return losses, f_dt, f_gl, m, weight_set, st_ims_mask
                    else:
                        return self.use_student_network(tc_images, targets_, return_maps=True, is_meanT=True)
                else:
                    loss_dict = {}
                    if domain == 'source':
                        domain_label = 0
                        ims_mask_st1 = to_image_list(ims_mask_st1).tensors
                        for idx in range(len(self.used_feature_layers)):
                            layer = "P" + str(self.used_feature_layers[idx])
                            if self.USE_DIS_DETECT_GL:
                                mode_ddis = getattr(self, 'ddis'+str(idx+3))
                                loss_dict["loss_detect_%s_ds" % layer] = \
                                    self.dt_dis_lambda * mode_ddis(features_lc_s[layer], domain_label, domain=domain, ims_mask=ims_mask_st1)
                            if self.USE_DIS_GLOBAL:
                                mode_fdis = getattr(self, 'fdis'+str(idx+3))
                                loss_dict["loss_adv_%s_ds" % layer] = \
                                    self.ga_dis_lambda * mode_fdis(features_gl_s[layer], domain_label, domain=domain, ims_mask=ims_mask_st1)
                            if self.USE_CC_GLOBAL:
                                mode_dis_cc = getattr(self, 'dis_cc'+str(idx+3))
                                loss_dict["loss_cc_%s_ds" % layer] = \
                                    self.cm_dis_lambda * mode_dis_cc(features_gl_s[layer], domain_label, gt_weights, None, layer, domain=domain, ims_mask=ims_mask_st1)
                    else:
                        domain_label = 1
                        for idx in range(len(self.used_feature_layers)):
                            layer = "P" + str(self.used_feature_layers[idx])
                            if self.USE_DIS_DETECT_GL:
                                mode_ddis = getattr(self, 'ddis'+str(idx+3))
                                loss_dict["loss_detect_%s_dt" % layer] = \
                                    self.dt_dis_lambda * mode_ddis(features_lc_s[layer], domain_label, domain=domain, ims_mask=ims_mask_st1)
                            if self.USE_DIS_GLOBAL:
                                mode_fdis = getattr(self, 'fdis'+str(idx+3))
                                loss_dict["loss_adv_%s_dt" % layer] = \
                                    self.ga_dis_lambda * mode_fdis(features_gl_s[layer], domain_label, domain=domain, ims_mask=ims_mask_st1)
                            if self.USE_CC_GLOBAL:
                                mode_dis_cc = getattr(self, 'dis_cc'+str(idx+3))
                                loss_dict["loss_cc_%s_dt" % layer] = \
                                    self.cm_dis_lambda * mode_dis_cc(features_gl_s[layer], domain_label, gt_weights, None, layer, domain=domain, ims_mask=ims_mask_st1)
                        
                                
                    return loss_dict
            
        elif valid:
           if val_net == 0:
               return self.use_student_network(tc_images, targets_, return_maps=True)
           else:
               return self.use_teacher_network(tc_images, targets_, return_maps=True)
        else:
            if self.cfg.MODEL.MODE.TEST_PROCESS == 'TC':
                _, _, _, _, _, results = self.use_teacher_network(tc_images, None, return_maps=True)
                return results
            else:
                _, _, _, _, _, results =self.use_student_network(tc_images, None, return_maps=True)
                return results

    
    def generate_target(self, dets):
        return dets
    
    def tranformer_target(self, st1_images, targets_, st_trans, ims_mask_st1=None, st2_images=None, ims_mask_st2=None, vis=False):
        lst1_images = to_image_list(st1_images).tensors
        ims_mask_st1 = to_image_list(ims_mask_st1).tensors
        lst2_images = to_image_list(st2_images).tensors
        ims_mask_st2 = to_image_list(ims_mask_st2).tensors
        
        targetn_ = []
        stn_images = []
        nimg_metas = []
        nims_mask = []
        img_num = len(targets_)
        det_bboxes = []
        det_labels = []
        for idx in range(img_num):
            target_ = targets_[idx]
            st_tran = st_trans[idx]
            
            bbs = target_.bbox.clone().detach()
            lls = target_.get_field('labels').clone().detach()
            min_hw = 3
            if 'RandomSized' in st_tran.keys():
                 bbs[:, 0] *= st_tran['RandomSized'][0]
                 bbs[:, 1] *= st_tran['RandomSized'][1]
                 bbs[:, 2] *= st_tran['RandomSized'][0]
                 bbs[:, 3] *= st_tran['RandomSized'][1]
            
            if 'CenterCrop' in st_tran.keys():
                 x00 = bbs[:, 0] - st_tran['CenterCrop'][0]
                 y00 = bbs[:, 1] - st_tran['CenterCrop'][1]
                 x01 = bbs[:, 2] - st_tran['CenterCrop'][0]
                 y01 = bbs[:, 3] - st_tran['CenterCrop'][1]
                 
                 mask0 = (x01 > min_hw) & (y01 > min_hw)
                 
                 x10 = st_tran['CenterCrop'][4] - x00
                 y10 = st_tran['CenterCrop'][5] - y00
                 x11 = st_tran['CenterCrop'][4] - x01
                 y11 = st_tran['CenterCrop'][5] - y01
                 
                 mask1 = (x10 > min_hw) & (y10 > min_hw)
                 
                 final_mask = mask0 & mask1
                 nbbs = torch.stack((x00, y00, x01, y01), dim=1)
                 bbs = nbbs[final_mask, :]
                 lls = lls[final_mask]
                 
                 im_bbs = torch.zeros_like(bbs)
                 im_bbs[:,0] = 0
                 im_bbs[:,1] = 0
                 im_bbs[:,2] = st_tran['CenterCrop'][4]
                 im_bbs[:,3] = st_tran['CenterCrop'][5]
                 
                 ##gt iou
                 gt_iou_ww = bbs[:,2] - bbs[:,0]
                 gt_iou_hh = bbs[:,3] - bbs[:,1]
                 gt_iou_ww[gt_iou_ww<0] = 0
                 gt_iou_hh[gt_iou_hh<0] = 0
                 gt_gl_areas = gt_iou_ww.mul(gt_iou_hh)
                 
                 gt_com_x0= torch.where((bbs[:, 0] - im_bbs[:, 0])>0, bbs[:, 0], im_bbs[:, 0])
                 gt_com_x1 = torch.where((bbs[:, 2] - im_bbs[:, 2])<0, bbs[:, 2], im_bbs[:, 2])
                 
                 gt_com_y0= torch.where((bbs[:, 1] - im_bbs[:, 1])>0, bbs[:, 1], im_bbs[:, 1])
                 gt_com_y1 = torch.where((bbs[:, 3] - im_bbs[:, 3])<0, bbs[:, 3], im_bbs[:, 3])
                 
                 
                 gt_com_ww = gt_com_x1 - gt_com_x0
                 gt_com_hh = gt_com_y1 - gt_com_y0
                 gt_com_areas = gt_com_ww.mul(gt_com_hh)
                 gt_iou = gt_com_areas.div(gt_gl_areas + 1e-5)
                 iou_mask = gt_iou > 0.5
                 
                 bbs = bbs[iou_mask,:]
                 lls = lls[iou_mask]
                 
                 
                 bb2 = bbs[:, 2]
                 bb3 = bbs[:, 3]
                 bb2[bb2>=st_tran['CenterCrop'][4]] = st_tran['CenterCrop'][4] - 1
                 bb3[bb3>=st_tran['CenterCrop'][5]] = st_tran['CenterCrop'][5] - 1
                 bbs[:, 2] = bb2
                 bbs[:, 3] = bb3
                 bbs[bbs<0] = 0.0
                 
                 sww = bbs[:, 2] - bbs[:, 0]
                 shh = bbs[:, 3] - bbs[:, 1]
                 
                 mask2 = (sww > min_hw) & (sww < 0.8 * st_tran['CenterCrop'][4])
                 mask3 = (shh > min_hw) & (shh < 0.8 * st_tran['CenterCrop'][5])
                 sz_mask = mask2 & mask3
                 bbs = bbs[sz_mask,:]
                 lls = lls[sz_mask]
            
            
            if 'RHF' in st_tran.keys():
                x0 = st_tran['RHF'][0] - bbs[:, 2]
                x1 = st_tran['RHF'][0] - bbs[:, 0]
                bbs[:, 0] = x0
                bbs[:, 2] = x1
            
            if 'RVF' in st_tran.keys():
                y0 = st_tran['RVF'][1] - bbs[:, 3]
                y1 = st_tran['RVF'][1] - bbs[:, 1]
                bbs[:, 1] = y0
                bbs[:, 3] = y1
            
            if len(bbs.size()) == 2 and bbs.size(0) > 0:
               h, w = st1_images.image_sizes[idx]
               boxlist = BoxList(bbs, (int(w), int(h)), mode="xyxy")
               boxlist.add_field("labels", lls)
               stn_images.append(lst1_images[idx])
               targetn_.append(boxlist)
               if ims_mask_st1 is not None:
                   nims_mask.append(ims_mask_st1[idx])
               
               if vis:
                   lim = lst1_images[idx].permute(1,2,0).contiguous()
                   lim -= lim.min()
                   cv_im = lim.cpu().detach().numpy()
                   cv_im = np.array(cv_im, np.int32)
                   
                   mbbs = bbs.cpu().detach().numpy()
                   for ii in range(bbs.size(0)):
                       cv2.rectangle(cv_im, (int(mbbs[ii,0]), int(mbbs[ii,1])), (int(mbbs[ii,2]), int(mbbs[ii,3])), (0,0,255), 2)
                   
                   cv2.imwrite('./test'+str(idx)+'.jpg', cv_im)
               
            elif target_.bbox.size(0) > 0:
               h, w = st2_images.image_sizes[idx]
               boxlist = BoxList(targets_[idx].bbox.clone().detach(), (int(w), int(h)), mode="xyxy")
               boxlist.add_field("labels", targets_[idx].get_field('labels').clone().detach())
               targetn_.append(boxlist)
               
               stn_images.append(lst2_images[idx])
               if ims_mask_st2 is not None:
                   nims_mask.append(ims_mask_st2[idx])
               
        stn_images = torch.stack(stn_images, dim=0).clone().detach()
        if ims_mask_st1 is not None:
            nims_mask = torch.stack(nims_mask, dim=0).clone().detach()
        return stn_images, targetn_, nims_mask
    
        
    def update_teacher(self, NET_MOMENTUM=None):
        if NET_MOMENTUM is None:
            NET_MOMENTUM = self.NET_MOMENTUM
            
        for param_diff, param_mn in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_mn.data = param_mn.data.clone() * NET_MOMENTUM + param_diff.data.clone() * (1. - NET_MOMENTUM)
        
        if self.use_gate:
            for param_diff, param_mn in zip(self.student_genbox.parameters(), self.teacher_genbox.parameters()):
                param_mn.data = param_mn.data.clone() * NET_MOMENTUM + param_diff.data.clone() * (1. - NET_MOMENTUM)
            
            for param_diff, param_mn in zip(self.student_genfeature.parameters(), self.teacher_genfeature.parameters()):
                param_mn.data = param_mn.data.clone() * NET_MOMENTUM + param_diff.data.clone() * (1. - NET_MOMENTUM)
        
        for param_diff, param_mn in zip(self.student_fcos.parameters(), self.teacher_fcos.parameters()):
            param_mn.data = param_mn.data.clone() * NET_MOMENTUM + param_diff.data.clone() * (1. - NET_MOMENTUM)
        
        return
    
    def update_student(self, NET_MOMENTUM=None):
        if NET_MOMENTUM is None:
            NET_MOMENTUM = self.NET_MOMENTUM
            
        for param_diff, param_mn in zip(self.teacher_backbone.parameters(), self.student_backbone.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        if self.use_gate:
            for param_diff, param_mn in zip(self.teacher_genbox.parameters(), self.student_genbox.parameters()):
                param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
            
            for param_diff, param_mn in zip(self.teacher_genfeature.parameters(), self.student_genfeature.parameters()):
                param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        for param_diff, param_mn in zip(self.teacher_fcos.parameters(), self.student_fcos.parameters()):
            param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
        
        return
    
    def update_instNet(self):
        NET_MOMENTUM = 0.9
        for idx in range(1):
            tc_instance_net = getattr(self, 'tc_instance_net'+str(idx+2))
            st_instance_net = getattr(self, 'instance_net'+str(idx+2))
            for param_diff, param_mn in zip(st_instance_net.parameters(), tc_instance_net.parameters()):
                param_mn.data = param_mn.data.clone() * (1. - NET_MOMENTUM) + param_diff.data.clone() * NET_MOMENTUM
    
        return
    
        
