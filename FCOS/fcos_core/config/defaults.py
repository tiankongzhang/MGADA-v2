# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as DCN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_CC = DCN()

_CC.MODEL = DCN()
_CC.MODEL.RPN_ONLY = False
_CC.MODEL.MASK_ON = False
_CC.MODEL.FCOS_ON = True
_CC.MODEL.DA_ON = True
_CC.MODEL.RETINANET_ON = False
_CC.MODEL.KEYPOINT_ON = False
_CC.MODEL.DEVICE = "cuda"
_CC.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_CC.MODEL.CLS_AGNOSTIC_BBOX_REG = False
_CC.MODEL.ISSAMPLE = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_CC.MODEL.WEIGHT = ""
_CC.MODEL.PRETAINED = ""
_CC.MODEL.USE_SYNCBN = False

_CC.MODEL.MODE = DCN()
_CC.MODEL.MODE.USE_STUDENT = False
_CC.MODEL.MODE.INSRANCE_ADV = False
_CC.MODEL.MODE.UPDATE_TEACHER_ITERUM = 100
_CC.MODEL.MODE.NET_MOMENTUM = 0.99
_CC.MODEL.MODE.INIT_DIS_NET = 100

_CC.MODEL.MODE.UPDATE_TEACHER_MARK = False
_CC.MODEL.MODE.UPDATE_STUDENT_MARK = False
_CC.MODEL.MODE.UPDATE_STUDENT_ITERUM = 20
_CC.MODEL.MODE.UPDATE_TEACHER_LR = [0.99, 0.92, 0.99, 0.99, 0.99]
_CC.MODEL.MODE.UPDATE_TEACHER_INTER = [1, 100, 1, 100, 100]

_CC.MODEL.MODE.TRAIN_PROCESS = 'S0'
_CC.MODEL.MODE.TEST_PROCESS = 'ST'
_CC.MODEL.MODE.USED_FEATURE_LAYERS = [3,4,5,6,7]

_CC.MODEL.MODE.USE_SL_TRAINING = False
_CC.MODEL.MODE.USE_AEMA = False
_CC.MODEL.MODE.USE_TEACHER = False
_CC.MODEL.MODE.USE_GATE = False
_CC.MODEL.MODE.REINIT_STUDENT_MARK = False
_CC.MODEL.MODE.SL_TRAINING_START = 0
_CC.MODEL.MODE.GATE_FUSION = 0# gate:0, conv:1, ave: 2
_CC.MODEL.MODE.WEIGHT_LOSS_SOURCE = 1.0
_CC.MODEL.MODE.WEIGHT_LOSS_TARGET = 1.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_CC.INPUT = DCN()
# Size of the smallest side of the image during training
_CC.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# The range of the smallest side for multi-scale training
_CC.INPUT.MIN_SIZE_RANGE_TRAIN = (-1, -1)  # -1 means disabled and it will use MIN_SIZE_TRAIN
# Maximum size of the side of the image during training
_CC.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_CC.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_CC.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_CC.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_CC.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_CC.INPUT.TO_BGR255 = True


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CC.DATASETS = DCN()
# # List of the dataset names for training, as present in paths_catalog.py
# _CC.DATASETS.TRAIN = ()
# List of the dataset names for training of source domain, as present in paths_catalog.py
_CC.DATASETS.TRAIN_SOURCE = ()
# List of the dataset names for training of target domain, as present in paths_catalog.py
_CC.DATASETS.TRAIN_TARGET = ()
# List of the dataset names for testing, as present in paths_catalog.py
_CC.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_CC.DATALOADER = DCN()
# Number of data loading threads
_CC.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_CC.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_CC.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_CC.MODEL.BACKBONE = DCN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_CC.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_CC.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
# GN for backbone
_CC.MODEL.BACKBONE.USE_GN = False


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_CC.MODEL.FPN = DCN()
_CC.MODEL.FPN.USE_GN = False
_CC.MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_CC.MODEL.GROUP_NORM = DCN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_CC.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_CC.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_CC.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_CC.MODEL.RPN = DCN()
_CC.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_CC.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_CC.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_CC.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_CC.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_CC.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_CC.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_CC.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_CC.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_CC.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_CC.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_CC.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_CC.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_CC.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_CC.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_CC.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_CC.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Custom rpn head, empty to use default conv or separable conv
_CC.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_CC.MODEL.ROI_HEADS = DCN()
_CC.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_CC.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_CC.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_CC.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_CC.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_CC.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_CC.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_CC.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_CC.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_CC.MODEL.ROI_BOX_HEAD = DCN()
_CC.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_CC.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_CC.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_CC.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_CC.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_CC.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
_CC.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_CC.MODEL.ROI_BOX_HEAD.DILATION = 1
_CC.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_CC.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4


_CC.MODEL.ROI_MASK_HEAD = DCN()
_CC.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_CC.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_CC.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_CC.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_CC.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_CC.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_CC.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_CC.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_CC.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_CC.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_CC.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_CC.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_CC.MODEL.ROI_MASK_HEAD.USE_GN = False

_CC.MODEL.ROI_KEYPOINT_HEAD = DCN()
_CC.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_CC.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_CC.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_CC.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_CC.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_CC.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_CC.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_CC.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_CC.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_CC.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_CC.MODEL.RESNETS = DCN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_CC.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_CC.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_CC.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_CC.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_CC.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_CC.MODEL.RESNETS.RES5_DILATION = 1

_CC.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_CC.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_CC.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# ---------------------------------------------------------------------------- #
# GENBOX Options
# ---------------------------------------------------------------------------- #
_CC.MODEL.GENBOX = DCN()
#_CC.MODEL.GENBOX.USE_GENBOX = False
_CC.MODEL.GENBOX.FPN_STRIDES = [8, 16, 32, 64, 128]
_CC.MODEL.GENBOX.NUM = 5

# the number of convolutions used in the cls and bbox tower
_CC.MODEL.GENBOX.NUM_CONVS = 4

# ---------------------------------------------------------------------------- #
# GENFEATURE Options
# ---------------------------------------------------------------------------- #
_CC.MODEL.GENFEATURE = DCN()
#_CC.MODEL.GENFEATURE.USE_GENFEATURE = False
_CC.MODEL.GENFEATURE.TWOMULTSCALE = True
_CC.MODEL.GENFEATURE.LOCAL_GLOBAL_MERGE = True
_CC.MODEL.GENFEATURE.POOLER_SCALES = [1.0 / 8, 1.0/16, 1.0/32, 1.0/64, 1.0/128]
_CC.MODEL.GENFEATURE.POOLER_RESOLUTION = 7
_CC.MODEL.GENFEATURE.CHANNELS_LEVEL = [64, 128, 256, 512, 1024]

# Focal loss parameter: alpha
_CC.MODEL.GENFEATURE.POOLER_SAMPLING_RATIO = 2
_CC.MODEL.GENFEATURE.FPN_STRIDES = [8, 16, 32, 64, 128]

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_CC.MODEL.FCOS = DCN()
_CC.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
_CC.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_CC.MODEL.FCOS.PRIOR_PROB = 0.01
_CC.MODEL.FCOS.INFERENCE_TH = 0.05
_CC.MODEL.FCOS.NMS_TH = 0.6
_CC.MODEL.FCOS.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
_CC.MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_CC.MODEL.FCOS.LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
_CC.MODEL.FCOS.NUM_CONVS = 4

# ---------------------------------------------------------------------------- #
# Domain Adaption Options
# ---------------------------------------------------------------------------- #
_CC.MODEL.DETECT = DCN()

# global and center-aware alignment
_CC.MODEL.DETECT.USE_DIS_GLOBAL = False
_CC.MODEL.DETECT.DT_DIS_LAMBDA = 0.01
_CC.MODEL.DETECT.GRL_APPLIED_DOMAIN = "both"

# the number of convolutions used in the discriminator
_CC.MODEL.DETECT.DIS_NUM_CONVS = 4

# adversarial parameter: GRL weight for global discriminator
_CC.MODEL.DETECT.GRL_WEIGHT_PL = 0.1

_CC.MODEL.CM = DCN()
# discriminator of different feature layers
_CC.MODEL.CM.USE_DIS_P7 = False
_CC.MODEL.CM.USE_DIS_P6 = False
_CC.MODEL.CM.USE_DIS_P5 = False
_CC.MODEL.CM.USE_DIS_P4 = False
_CC.MODEL.CM.USE_DIS_P3 = False

# global and center-aware alignment
_CC.MODEL.CM.USE_CM_GLOBAL = False
_CC.MODEL.CM.GL_CM_LAMBDA = 0.01
_CC.MODEL.CM.GRL_APPLIED_DOMAIN = "both"

# the number of convolutions used in the discriminator
_CC.MODEL.CM.DIS_NUM_CONVS = 4

# adversarial parameter: GRL weight for global discriminator
_CC.MODEL.CM.GRL_WEIGHT_PL = 0.1

_CC.MODEL.CM.LOSS_DIRECT_W = 0.01
_CC.MODEL.CM.LOSS_GRL_W = 0.01
_CC.MODEL.CM.SAMPLES_THRESH = 0.5

_CC.MODEL.ADV = DCN()

# discriminator of different feature layers
_CC.MODEL.ADV.USE_DIS_P7 = False
_CC.MODEL.ADV.USE_DIS_P6 = False
_CC.MODEL.ADV.USE_DIS_P5 = False
_CC.MODEL.ADV.USE_DIS_P4 = False
_CC.MODEL.ADV.USE_DIS_P3 = False

# global and center-aware alignment
_CC.MODEL.ADV.USE_DIS_GLOBAL = False
_CC.MODEL.ADV.USE_DIS_CENTER_AWARE = False
_CC.MODEL.ADV.CENTER_AWARE_WEIGHT = 20
_CC.MODEL.ADV.CENTER_AWARE_TYPE = "ca_feature"
_CC.MODEL.ADV.GA_DIS_LAMBDA = 0.01
_CC.MODEL.ADV.CA_DIS_LAMBDA = 0.1
_CC.MODEL.ADV.GRL_APPLIED_DOMAIN = "both"

# the number of convolutions used in the discriminator
_CC.MODEL.ADV.DIS_NUM_CONVS = 4

# the number of convolutions used in the center-aware discriminator
_CC.MODEL.ADV.CA_DIS_NUM_CONVS = 4

# adversarial parameter: GRL weight for global discriminator
_CC.MODEL.ADV.GRL_WEIGHT_PL = 0.1

# adversarial parameter: GRL weight for center-aware discriminator
_CC.MODEL.ADV.CA_GRL_WEIGHT_PL = 0.1


# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_CC.MODEL.RETINANET = DCN()

# This is the number of foreground classes and background.
_CC.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_CC.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_CC.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_CC.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_CC.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_CC.MODEL.RETINANET.OCTAVE = 2.0
_CC.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_CC.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_CC.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_CC.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_CC.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_CC.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_CC.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_CC.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_CC.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_CC.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_CC.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_CC.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_CC.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_CC.MODEL.FBNET = DCN()
_CC.MODEL.FBNET.ARCH = "default"
# custom arch
_CC.MODEL.FBNET.ARCH_DEF = ""
_CC.MODEL.FBNET.BN_TYPE = "bn"
_CC.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_CC.MODEL.FBNET.WIDTH_DIVISOR = 1
_CC.MODEL.FBNET.DW_CONV_SKIP_BN = True
_CC.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_CC.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_CC.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_CC.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_CC.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_CC.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_CC.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_CC.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_CC.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_CC.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_CC.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_CC.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CC.SOLVER = DCN()
_CC.SOLVER.MAX_ITER = 40000

# _CC.SOLVER.BASE_LR = 0.001
# _CC.SOLVER.BIAS_LR_FACTOR = 2

_CC.SOLVER.MOMENTUM = 0.9

_CC.SOLVER.WEIGHT_DECAY = 0.0005
_CC.SOLVER.WEIGHT_DECAY_BIAS = 0

# _CC.SOLVER.GAMMA = 0.1
# _CC.SOLVER.STEPS = (30000,)

# _CC.SOLVER.WARMUP_FACTOR = 1.0 / 3
# _CC.SOLVER.WARMUP_ITERS = 500
# _CC.SOLVER.WARMUP_METHOD = "linear"

_CC.SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_CC.SOLVER.IMS_PER_BATCH = 16

# Backbone
_CC.SOLVER.BACKBONE = DCN()
_CC.SOLVER.BACKBONE.BASE_LR = 0.005
_CC.SOLVER.BACKBONE.BIAS_LR_FACTOR = 2
_CC.SOLVER.BACKBONE.GAMMA = 0.1
_CC.SOLVER.BACKBONE.STEPS = (30000,)
_CC.SOLVER.BACKBONE.WARMUP_FACTOR = 1.0 / 3
_CC.SOLVER.BACKBONE.WARMUP_ITERS = 500
_CC.SOLVER.BACKBONE.WARMUP_METHOD = "linear"
# GENBOX
_CC.SOLVER.GENBOX = DCN()
_CC.SOLVER.GENBOX.BASE_LR = 0.005
_CC.SOLVER.GENBOX.BIAS_LR_FACTOR = 2
_CC.SOLVER.GENBOX.GAMMA = 0.1
_CC.SOLVER.GENBOX.STEPS = (30000,)
_CC.SOLVER.GENBOX.WARMUP_FACTOR = 1.0 / 3
_CC.SOLVER.GENBOX.WARMUP_ITERS = 500
_CC.SOLVER.GENBOX.WARMUP_METHOD = "linear"
# GENFEATURE
_CC.SOLVER.GENFEATURE = DCN()
_CC.SOLVER.GENFEATURE.BASE_LR = 0.005
_CC.SOLVER.GENFEATURE.BIAS_LR_FACTOR = 2
_CC.SOLVER.GENFEATURE.GAMMA = 0.1
_CC.SOLVER.GENFEATURE.STEPS = (30000,)
_CC.SOLVER.GENFEATURE.WARMUP_FACTOR = 1.0 / 3
_CC.SOLVER.GENFEATURE.WARMUP_ITERS = 500
_CC.SOLVER.GENFEATURE.WARMUP_METHOD = "linear"
# FCOS
_CC.SOLVER.FCOS = DCN()
_CC.SOLVER.FCOS.BASE_LR = 0.005
_CC.SOLVER.FCOS.BIAS_LR_FACTOR = 2
_CC.SOLVER.FCOS.GAMMA = 0.1
_CC.SOLVER.FCOS.STEPS = (30000,)
_CC.SOLVER.FCOS.WARMUP_FACTOR = 1.0 / 3
_CC.SOLVER.FCOS.WARMUP_ITERS = 500
_CC.SOLVER.FCOS.WARMUP_METHOD = "linear"
# Discriminator
_CC.SOLVER.DIS = DCN()
_CC.SOLVER.DIS.BASE_LR = 0.005
_CC.SOLVER.DIS.BIAS_LR_FACTOR = 2
_CC.SOLVER.DIS.GAMMA = 0.1
_CC.SOLVER.DIS.STEPS = (30000,)
_CC.SOLVER.DIS.WARMUP_FACTOR = 1.0 / 3
_CC.SOLVER.DIS.WARMUP_ITERS = 500
_CC.SOLVER.DIS.WARMUP_METHOD = "linear"
# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_CC.TEST = DCN()
_CC.TEST.EXPECTED_RESULTS = []
_CC.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_CC.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_CC.TEST.DETECTIONS_PER_IMG = 100


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_CC.OUTPUT_DIR = "."

_CC.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
