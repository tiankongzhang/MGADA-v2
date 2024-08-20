# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from fcos_core.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        
        images_tc = []
        metas_tc = []
        ims_mask_tc = []
        images_st = []
        metas_st = []
        ims_mask_st = []
        for idx in range(len(transposed_batch[3])):
             images_tc.append(transposed_batch[3][idx]['sc_img'])
             metas_tc.append(transposed_batch[3][idx]['sc_tran_dict'])
             ims_mask_tc.append(transposed_batch[3][idx]['img_mask_sc'])
             images_st.append(transposed_batch[3][idx]['tr_img'])
             metas_st.append(transposed_batch[3][idx]['tr_tran_dict'])
             ims_mask_st.append(transposed_batch[3][idx]['img_mask_tr'])
             
        images_tc = to_image_list(images_tc, self.size_divisible)
        ims_mask_tc = to_image_list(ims_mask_tc, self.size_divisible)
        images_st = to_image_list(images_st, self.size_divisible)
        ims_mask_st = to_image_list(ims_mask_st, self.size_divisible)
        return images, targets, img_ids, images_tc, metas_tc, ims_mask_tc, images_st, ims_mask_st
