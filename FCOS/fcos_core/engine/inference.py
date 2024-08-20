# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .visualizer import Visualizer


def compute_on_dataset(model, data_loader, device, timer=None):

    # model.eval
    #for k in model:
    #    model[k].eval()
    model.eval()
    
    results_dict = {}
    cpu_device = torch.device("cpu")
    
    img_index = 0
    colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
        [255, 100, 100],
        [100, 255, 100],
        [100, 100, 255]
    ]
    
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids, _, _,_, _, _ = batch
        #images, images_o, image_ids, _, _,_, _, _ = batch
        #print(len(images_o), image_ids)
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            #output = foward_detector(model, images, targets=None)
            #output = model(images, None, domain='s')
            output = model(images, None, domain='s')
            '''
            ##draw
            colors_edge = [
                [1.0, 1.0, 1.0],
                [0.000, 1.000, 1.000],
                [1.000, 1.000, 0.000],
                [0.333, 0.333, 1.000],
                [0.929, 0.694, 0.125],
                [0.000, 1.000, 0.000],
                [0.000, 0.333, 0.500],
                [0.000, 0.000, 1.000],
                [0.667, 0.000, 1.000],
            ]
            
            colors_text = [
                [75, 0, 0],
                [0, 75, 0],
                [0, 0, 75],
                [255, 255, 50],
                [255, 50, 255],
                [50, 255, 255],
                [205, 255, 255],
                [255, 100, 150],
                [100, 255, 150],
                [150, 100, 255]
            ]
            category = {1: "person", 2:"car", 3:"train", 4:"rider", 5:"truck", 6: "mbike", 7: "bicycle", 8:"bus"}
            #category = {1: "car"}
            images_np = images.tensors.permute(0, 2, 3, 1).cpu().detach().numpy()
            for idx in range(len(output)):
                ###visual-other method
                #city, foggy
                #if image_ids[idx] not in [36, 182, 470, 438, 432, 21, 235, 359, 228]:
                #    continue
                
                #if image_ids[idx] not in [21, 228]:
                #    continue
                
                #kitti, sim10k
                #if image_ids[idx] not in [90, 17, 83, 162]:
                #    continue
                
                #if image_ids[idx] not in [162]:
                #    continue
                
                ###visual-ema and aema
                #if image_ids[idx] not in [182, 223, 184]:
                #     continue
                
                #if image_ids[idx] not in [182]:
                #     continue
                
                #if image_ids[idx] not in [184]:
                #     continue
                
                if image_ids[idx] not in [223]:
                     continue
                    
                bbs = output[idx].bbox.cpu().detach().numpy()
                labels_ = output[idx].get_field("labels").cpu().detach().numpy()
                scores_ = output[idx].get_field("scores").cpu().detach().numpy()
                #print(bbs.shape, labels.shape, scores.shape)
                
                vis = Visualizer(images_o[idx], metadata=None)
                colors = []
                labels = []
                nboxes = []
                
                #print(images_o[idx].size[0], images_o[idx].size[1], images_np[idx].shape)
                hh = images_o[idx].size[0] / images_np[idx].shape[1]
                ww = images_o[idx].size[1] / images_np[idx].shape[0]
                for idx_b in range(bbs.shape[0]):
                    ###visual-other method
                    #if str(image_ids[idx]) == "21" and (idx_b == 22 or idx_b == 8 or idx_b == 11):
                    #     continue
                    
                    #kitti, sim10k
                    #if str(image_ids[idx]) == "162" and idx_b == 5:
                    #     continue
                    
                    #if str(image_ids[idx]) == "162" and (idx_b == 10 or idx_b == 7 or idx_b == 6):
                    #     continue
                    
                    ###visual-ema and aema
                    #if str(image_ids[idx]) == "182" and idx_b == 2:
                    #     continue
                    
                    #if str(image_ids[idx]) == "184" and idx_b == 8:
                    #     continue
                    
                    #if str(image_ids[idx]) == "223" and idx_b == 12:
                    #     continue
                    
                    #if str(image_ids[idx]) == "184" and idx_b == 4:
                    #     continue
                    
                    #if str(image_ids[idx]) == "182" and idx_b == 7:
                    #     continue
                    
                    if str(image_ids[idx]) == "223" and idx_b == 23:
                         continue
                    
                    bb = bbs[idx_b]
                    x0 = int(bb[0]* ww)
                    y0 = int(bb[1]* hh)
                    x1 = int(bb[2]* ww)
                    y1 = int(bb[3]* hh)
                    
                    ####visual-other method
                    #if str(image_ids[idx]) == "21" and idx_b == 3:
                    #   x0 += 10
                    #   x1 += 10
                    #   y0 += 10
                    #   y1 += 10
                    
                    
                    score = scores_[idx_b]
                    clsll = labels_[idx_b]
                    #if score < 0.42:
                    #   score = 0.43
                    
                    if score < 0.42:
                       continue
                    
                    labels.append('{}:{:.0f}%'.format(category[clsll], score * 100))
                    #labels.append('{}:{:.0f}%'.format(str(idx_b), score * 100))
                    colors.append(colors_edge[clsll])
                    
                    #bb = boxes[idx_b]*1
                    #bb[2] = bb[0] + bb[2]
                    #bb[3] = bb[1] + bb[3]
                    nboxes.append([x0, y0, x1, y1])
                
                out = vis.overlay_instances(
                    boxes=nboxes,
                    labels=labels,
                    masks=None,
                    assigned_colors=colors,
                    alpha=0.8,
                )
                
                #out.save(os.path.join('./images_data', os.path.basename(str(image_ids[idx])+"_") + "MGA" + ".png"))
                #out.save(os.path.join('./images_data', os.path.basename(str(image_ids[idx])+"_") + "BS" + ".png"))
                
                #out.save(os.path.join('./images_data', os.path.basename(str(image_ids[idx])+"_") + "EMA" + ".png"))
                out.save(os.path.join('./images_data', os.path.basename(str(image_ids[idx])+"_") + "EMA_A2" + ".png"))
            '''
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
