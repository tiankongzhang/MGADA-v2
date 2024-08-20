# FCOS Core
Note that `fcos_core` corresponds to `maskrcnn_benchmark` in the original maskrcnn_benchmark repository. \
We changed the name to avoid conflicts with the original maskrcnn-benchmark installation.

The core code of FCOS detector is located under [modeling/rpn/fcos](modeling/rpn/fcos).

nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=2900 tools/train_net_db.py --config-file ./configs/v2/city/VGG16/ablation/s1_w_gate_algin_aema_st_com1/da_ga_cityscapes_VGG_16_FPN_4x-s1.yaml OUTPUT_DIR /data/Project/PAMI/Major/COM/method1/ MODEL.ISSAMPLE True > /data/Project/PAMI/Major/COM/method1/0000.log 2>&1 &
