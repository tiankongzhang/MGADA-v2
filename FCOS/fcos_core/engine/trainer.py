# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fcos_core.utils.comm import synchronize, get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

from fcos_core.structures.image_list import to_image_list
from fcos_core.engine.inference import inference
import torch.multiprocessing as mp
from fcos_core.modeling.momentum.MemoryManagement import MemoryManagement
from fcos_core.utils.comm import synchronize
from fcos_core.structures.bounding_box import BoxList

def IoU(target, pred, weight=None, eps=1e-6):
        pn, tn = pred.size(0), target.size(0)
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        target_aera = (target_right - target_left) * (target_bottom - target_top)

        x0_intersect = torch.max(pred_left.unsqueeze(0).expand(tn, pn), target_left.unsqueeze(1).expand(tn, pn)) 
        x1_intersect = torch.min(pred_right.unsqueeze(0).expand(tn, pn), target_right.unsqueeze(1).expand(tn, pn)) 
        y0_intersect = torch.max(pred_top.unsqueeze(0).expand(tn, pn), target_top.unsqueeze(1).expand(tn, pn)) 
        y1_intersect = torch.min(pred_bottom.unsqueeze(0).expand(tn, pn), target_bottom.unsqueeze(1).expand(tn, pn)) 
        w_intersect = x1_intersect - x0_intersect
        h_intersect = y0_intersect - y1_intersect
        w_intersect[w_intersect<0] = 0
        h_intersect[h_intersect<0] = 0
        area_intersect = w_intersect * h_intersect
        gious = area_intersect / (target_aera[:, None] + eps)
        return gious


###memory module
def write_read_f(cfg, queue_quit, queue_in, queue_sout, queue_tout):
    model_mm = model_mm = MemoryManagement(cfg.MODEL.FCOS.NUM_CLASSES-1)
    use_source = True
    use_tagret = False
    read_mark = 0
    
    while queue_quit.empty():
        try:
            if not queue_in.empty():
               outputs_w = queue_in.get()
               if outputs_w['mark'] == 1:
                    model_mm(images=outputs_w['images'], targets=outputs_w['targets'], domain=outputs_w['domain'], method='w')
                
               else:
                    read_mark = 1
            
            if read_mark == 1:
                if use_source and (not queue_sout.full()):
                   outputs_sr = model_mm(domain='s', method='r')
                   
                   if outputs_sr['images'] is not None:
                       outputs_sr.update({'end_mark': 0})
                   else:
                       print('---end_mark:',1)
                       outputs_sr.update({'end_mark': 1})
                    
                   queue_sout.put(outputs_sr)
                
                if use_tagret and (not queue_tout.full()):
                    outputs_tr = model_mm(domain='t', method='r')
                        
                    if outputs_tr['images'] is not None:
                        outputs_tr.update({'end_mark': 0})
                    else:
                        outputs_tr.update({'end_mark': 1})
                        
                    queue_tout.put(outputs_tr)
                
        except EOFError:
            # 当out_pipe接受不到输出的时候且输入被关闭的时候，会抛出EORFError，可以捕获并且退出子进程
            break
    print('process end!')

def update_momentum_params(model, queue_in, queue_sout, queue_tout, device):
    model.eval()
    fst_w = 0.0
    iter_num = 10
    input_mm = {
        'mark': 2
     }
    queue_in.put(input_mm)
    
    with torch.no_grad():
        iter_index = 0
        inter_count = 0
        while iter_index < iter_num:
            if queue_sout.empty():
                continue
                
            outputs_mm_s = queue_sout.get()
            if outputs_mm_s['end_mark'] == 1:
                 break
            
            msrc_input = outputs_mm_s['images']
            msrc_targets = outputs_mm_s['targets']
            
            fsrc_input = []
            fsrc_targets = []
            for idx in range(len(msrc_targets)):
                lmsrc_targets = msrc_targets[idx]
                if len(lmsrc_targets.bbox.size())==2 and lmsrc_targets.bbox.size(0) > 0:
                    fsrc_targets.append(lmsrc_targets.to(device))
                    fsrc_input.append(msrc_input[idx].to(device))
            
            if len(fsrc_targets) > 0:
                fsrc_input = torch.stack(fsrc_input, dim=0)
                loss_st, _, _, _, _, _ = model(fsrc_input, fsrc_targets, valid=True, val_net=0)
                det_loss_st = loss_st['loss_centerness'] + loss_st['loss_reg']
                    
                loss_tc, _, _, _, _, _ = model(fsrc_input, fsrc_targets, valid=True, val_net=1)
                det_loss_tc = loss_tc['loss_centerness'] + loss_tc['loss_reg']
                
                if det_loss_st.cpu().numpy() < 1e-4 or det_loss_tc.cpu().numpy() < 1e-4:
                    iter_index += 1
                    continue
                    
                st_w_ct = loss_st['loss_centerness'] / (loss_st['loss_centerness'] + loss_tc['loss_centerness'] + 1e-5)
                st_w_reg = loss_st['loss_reg'] / (loss_st['loss_reg'] + loss_tc['loss_reg'] + 1e-5)
                st_w = (st_w_ct + st_w_reg) / 2
                fst_w += st_w
                inter_count += 1
            iter_index += 1
        
    fst_w = fst_w /(inter_count + 1e-5)
    model.train()
    if inter_count == 0:
       f_w = 1.0
    else:
       lmask_w = (fst_w < 0.5).float()
       f1 = torch.exp(3.0 * (0.5 - fst_w) * lmask_w)
       f2 = torch.exp(3.0 * (0.5 - fst_w) * (1 - lmask_w))
       
       f2 = f2.clamp(min=0.1)
       f_w = f1.mul(lmask_w) + f2.mul(1 - lmask_w)

    return f_w


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        model,
        data_loader,
        adap_cfg,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        val_dict,
        arguments,
):

    # dataloader
    data_loader_source = data_loader["source"]
    data_loader_target = data_loader["target"]

    # Start training
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")

    model.train()
    if adap_cfg.MODEL.MODE.TRAIN_PROCESS == 'S1':
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    motum_inter = adap_cfg.MODEL.MODE.UPDATE_TEACHER_ITERUM
    net_momentum = 1 - adap_cfg.MODEL.MODE.NET_MOMENTUM
    
    use_sl_training = adap_cfg.MODEL.MODE.USE_SL_TRAINING
    update_teacher_mark = adap_cfg.MODEL.MODE.UPDATE_TEACHER_MARK
    use_aema = adap_cfg.MODEL.MODE.USE_AEMA
    train_process = adap_cfg.MODEL.MODE.TRAIN_PROCESS
    sl_training_start = adap_cfg.MODEL.MODE.SL_TRAINING_START
    
    source_weight = adap_cfg.MODEL.MODE.WEIGHT_LOSS_SOURCE
    target_weight = adap_cfg.MODEL.MODE.WEIGHT_LOSS_TARGET
    
    ##start memory process
    if train_process == 'S1' and use_aema and update_teacher_mark:
        mp.set_start_method('spawn')
        queue_quit = mp.Queue(1)
        queue_in = mp.Queue(10)
        queue_sout = mp.Queue(10)
        queue_tout = mp.Queue(10)
        
        ctx = mp.get_context('spawn')
        son_p = ctx.Process(target=write_read_f, args=(adap_cfg, queue_quit, queue_in, queue_sout, queue_tout))
        son_p.start()

    meters = MetricLogger(delimiter="  ")
    assert len(data_loader_source) == len(data_loader_target)
    max_iter = max(len(data_loader_source), len(data_loader_target))
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    global_index = -1
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, ((images_s, targets_s, _, images_st1, metas_st1, ims_mask_st1, images_st2, ims_mask_st2), (images_t, _, _, images_st1_t, metas_st1_t, ims_mask_st1_t, images_st2_t, ims_mask_st2_t)) in enumerate(zip(data_loader_source, data_loader_target), start_iter):

        data_time = time.time() - end
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            for k in scheduler:
                scheduler[k].step()
        
        # optimizer.zero_grad()
        for k in optimizer:
            optimizer[k].zero_grad()
        
        #iteration = iteration + 1
        if adap_cfg.MODEL.MODE.TRAIN_PROCESS == 'S0':
              images_s = images_s.to(device)
              targets_s = [target_s.to(device) for target_s in targets_s]
              
              loss_dict_s, features_lc_s, features_gl_s, score_maps_s, _, _ = model(images_s, targets_s, domain='source')
              det_loss_s = sum(list(loss_dict_s.values()))
              
              # reduce losses over all GPUs for logging purposes
              loss_dict_reduced = reduce_loss_dict(loss_dict_s)
              losses_reduced = sum(loss for loss in loss_dict_reduced.values())
              meters.update(loss_gs=losses_reduced, **loss_dict_reduced)

              det_loss_s.backward()
              
        elif adap_cfg.MODEL.MODE.TRAIN_PROCESS == 'S1':
            ##########################################################################
            ######################### (2):  source domain ############################
            ##########################################################################
            ##det loss
            images_s = images_s.to(device)
            images_st1 = images_st1.to(device)
            images_st2 = images_st2.to(device)
            targets_s = [target_s.to(device) for target_s in targets_s]
            ims_mask_st1 = ims_mask_st1.to(device)
            ims_mask_st2 = ims_mask_st2.to(device)
            
            loss_dict_s, features_lc_s, features_gl_s, score_maps_s, weigth_set_s, ims_masks_  = model(images_s, targets_s, domain='source', st1_images=images_st1, st1_trans=metas_st1, ims_mask_st1=ims_mask_st1, st2_images=images_st2, ims_mask_st2=ims_mask_st2)
            det_loss_s = sum(list(loss_dict_s.values()))
            det_loss_s.backward(retain_graph=True)
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_s_reduced = reduce_loss_dict(loss_dict_s)
            losses_s_reduced = sum(loss for loss in loss_dict_s_reduced.values())
            meters.update(loss_gs=losses_s_reduced, **loss_dict_s_reduced)
            del det_loss_s, loss_dict_s
            
            ##adv loss
            loss_dict_ds = model(None, domain='source', mode=1, features_lc_s=features_lc_s, features_gl_s=features_gl_s, gt_weights = weigth_set_s, ims_mask_st1=ims_masks_)
            losses = source_weight * sum(loss for loss in loss_dict_ds.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict_ds)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_ds=losses_reduced, **loss_dict_reduced)

            losses.backward()
            del loss_dict_ds, losses
            
            
            ##########################################################################
            ######################### (2):  target domain ############################
            ##########################################################################
            ##det loss
            images_t = images_t.to(device)
            images_st1_t = images_st1_t.to(device)
            ims_mask_st1_t = ims_mask_st1_t.to(device)
            images_st2_t = images_st2_t.to(device)
            ims_mask_st2_t = ims_mask_st2_t.to(device)

            with torch.no_grad():
                stsc_images, st_targets_, st_immasks_  = model(images_t, st1_images=images_st1_t, st1_trans=metas_st1_t, ims_mask_st1=ims_mask_st1_t, st2_images=images_st2_t, ims_mask_st2=ims_mask_st2_t)
            
            stsc_images, st_targets_, st_immasks_ = M_func(images_s, to_image_list(stsc_images), targets_s, st_targets_, st_immasks_)
            loss_dict_t, features_lc_t, features_gl_t, score_maps_t, weigth_set_t, _  = model(stsc_images, st_targets_, domain='target')
            
            if use_sl_training and iteration >= sl_training_start:
                det_loss_t = 0.1 * sum(list(loss_dict_t.values()))
            else:
                det_loss_t = 0.0 * sum(list(loss_dict_t.values()))
            det_loss_t.backward(retain_graph=True)
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_t_reduced = reduce_loss_dict(loss_dict_t)
            losses_t_reduced = sum(loss for loss in loss_dict_t_reduced.values())
            meters.update(loss_gt=losses_t_reduced, **loss_dict_t_reduced)
            del det_loss_t, loss_dict_t
            
            
            ##adv loss
            loss_dict_dt = model(None, domain='target', mode=1, features_lc_s=features_lc_t, features_gl_s=features_gl_t, gt_weights = weigth_set_t, ims_mask_st1=st_immasks_)
            losses = target_weight * sum(loss for loss in loss_dict_dt.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict_dt)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_dt=losses_reduced, **loss_dict_reduced)

            losses.backward()
            del loss_dict_dt, losses
            
            if use_aema and update_teacher_mark:
                if not queue_in.full():
                    ntargets_s = []
                    for target_idx in range(len(targets_s)):
                        if len(targets_s[target_idx].bbox.size()) and targets_s[target_idx].bbox.size(0) > 0:
                            ntargets_s.append(targets_s[target_idx])
                    input_mm = {'images': to_image_list(images_s).tensors.clone().detach(), 'targets':targets_s,
                        'device': None,
                        'domain': 's',
                        'mark': 1
                     }
                    queue_in.put(input_mm)
            

        ##########################################################################
        ##########################################################################
        ##########################################################################
        for k in optimizer:
            optimizer[k].step()
        
        if pytorch_1_1_0_or_later:
            for k in scheduler:
                scheduler[k].step()
        
        if update_teacher_mark and iteration > 100 and iteration %  motum_inter == 0:
            if use_aema:
                gamma = update_momentum_params(model, queue_in, queue_sout, queue_tout, device)
                lnet_momentum = 1 - net_momentum * gamma
            else:
                 
                lnet_momentum = 1 - net_momentum
            model.module.update_teacher(NET_MOMENTUM=lnet_momentum)
        
        # End of training
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        
        if iteration > 10 and iteration % 500 == 0:
             model.eval()
             output_folders = val_dict['output_folders']
             dataset_names = val_dict['dataset_names']
             iou_types = val_dict['iou_types']
             box_only = val_dict['box_only']
             expected_results = val_dict['expected_results']
             expected_results_sigma_tol = val_dict['expected_results_sigma_tol']
             device = val_dict['device']
             for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loader["val"]):
                 inference(
                     model,
                     data_loader_val,
                     dataset_name=dataset_name,
                     iou_types=iou_types,
                     box_only=box_only,
                     device=device,
                     expected_results=expected_results,
                     expected_results_sigma_tol=expected_results_sigma_tol,
                     output_folder=output_folder,
                 )
                 synchronize()
                 
             model.train()
        
        
        #sample_layer = used_feature_layers[0]  # sample any one of used feature layer
        if iteration % 20 == 0 or iteration == max_iter:
            if 'student_backbone' in optimizer.keys():
                backbone_lr = optimizer["student_backbone"].param_groups[0]["lr"]
                fcos_lr = optimizer["student_fcos"].param_groups[0]["lr"]
            else:
                backbone_lr = optimizer["teacher_backbone"].param_groups[0]["lr"]
                fcos_lr = optimizer["teacher_fcos"].param_groups[0]["lr"]
            logger.info(
                meters.delimiter.join([
                    "eta: {eta}",
                    "iter: {iter}",
                    "{meters}",
                    "lr_backbone: {lr_backbone:.6f}",
                    "lr_fcos: {lr_fcos:.6f}",
                    "max mem: {memory:.0f}",
                ]).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr_backbone=backbone_lr,
                    lr_fcos=fcos_lr,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                ))
                

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
    
    queue_quit.put(1)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)))
