# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

"""

"""

# ------------------------------------------------
VERSION = 'new_tfrecord_test'
NET_NAME = 'resnet152_v1d'
DATASET_NAME = 'DOTA'
TFRECORD_PATH = '/content/drive/MyDrive/Colab_Notebooks/MyGITs/FPN_Tensorflow_Rotation/data/tfrecord/'

CLASS_NUM = 15

ROOT_PATH = '/content/drive/MyDrive/Colab_Notebooks/MyGITs/FPN_Tensorflow_Rotation/'

ROTATE_NMS_USE_GPU = True  # When Train, use GPU NMS, When Test, Use CPU NMS.

ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
#ROOT_PATH = os.path.abspath('../')
print (20*"++--")
print ('The ROOT_PATH is ',ROOT_PATH)
GPU_GROUP = "0"
NUM_GPU = len(GPU_GROUP.strip().split(','))
SHOW_TRAIN_INFO_INTE = 50 # prints info every 50 iterations; one image per iteration
SMRY_ITER = 500 # for training history in Tensorboard I think
SAVE_WEIGHTS_INTE = 20000 # the frequency at which to save weights. 1 image per iteration.  

SUMMARY_PATH = ROOT_PATH +'/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = False
FREEZE_BLOCKS = [True, True, False, False, False]  # for gluoncv backbone
FIXED_BLOCKS = 0  # allow 0~3
USE_07_METRIC = True
CUDA9 = True

RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0
FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0

MUTILPY_BIAS_GRADIENT = 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0  # if None, will not clip

BATCH_SIZE = 1 # keep as 1
EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001 * BATCH_SIZE * NUM_GPU
DECAY_STEP = [SAVE_WEIGHTS_INTE*12, SAVE_WEIGHTS_INTE*16, SAVE_WEIGHTS_INTE*20]
# DECAY_STEP = [SAVE_WEIGHTS_INTE*20,SAVE_WEIGHTS_INTE*24,SAVE_WEIGHTS_INTE*28]
MAX_ITERATION = SAVE_WEIGHTS_INTE*25
WARM_SETP = int(1.0 / 4.0 * SAVE_WEIGHTS_INTE)

# -------------------------------------------- Data_preprocess_config
#DATASET_NAME = 'DOTA'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [108.108, 108.108, 108.108]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_MEAN_ = [0.424, 0.424, 0.424] # used for resnet*_v1d net_name
#PIXEL_MEAN_ = [0.324, 0.424, 0.524] # used for resnet*_v1d net_name
PIXEL_STD = [0.23, 0.23, 0.23] # used for resnet*_v1d net_name

# IMG_SHORT_SIDE_LEN = 800
# IMG_SHORT_SIDE_LEN = [800, 900, 1000, 1100, 600, 400]
# IMG_MAX_LENGTH = 800
IMG_SHORT_SIDE_LEN = [1000, 1000, 1200, 800]
IMG_MAX_LENGTH = 2000 # if the cropped image dimensions exceed this value, the image will be resized. Recommend setting your crop size in data_crop.py to be equal to this value to prevent resizing all images at training time. 


## Data augmentation
IMG_ROTATE = True
RGB2GRAY = False
VERTICAL_FLIP = True
HORIZONTAL_FLIP = True
IMAGE_PYRAMID = True # Will randomly resize image dimensions using the values in IMG_SHORT_SIDE_LEN. Set to False if IMG_SHORT_SIDE_LEN is set to a single value. Using Image Pyramid is desiegned to help prevent overfitting and increase data augmentation. 

# --------------------------------------------- Network_config
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001

# ---------------------------------------------Anchor config
USE_CENTER_OFFSET = False

LEVLES = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]  # addjust the base anchor size for voc.
ANCHOR_STRIDE_LIST = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.0]
ANCHOR_RATIOS = [0.5, 1., 2.0, 1/4.0, 4.0, 1/6.0, 6.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0, 2.0]
ANCHOR_SCALE_FACTORS = None

# --------------------------------------------FPN config
SHARE_HEADS = True
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 512  # 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7  # 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000
RPN_MAXIMUM_PROPOSAL_TEST = 1000

# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.6  # only show in tensorboard

SOFT_NMS = False
FAST_RCNN_NMS_IOU_THRESHOLD = 0.3  # 0.5
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 200
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 512  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False

# -------------------------------------------mask config
USE_SUPERVISED_MASK = True
MASK_TYPE = 'r'  # r or h
BINARY_MASK = False
SIGMOID_ON_DOT = False
MASK_ACT_FET = True  # weather use mask generate 256 channels to dot feat.
GENERATE_MASK_LIST = ["P2", "P3", "P4", "P5"]
ADDITION_LAYERS = [4, 4, 4, 4]  # add 4 layer to generate P2_mask, 2 layer to generate P3_mask
ENLAEGE_RF_LIST = ["P2", "P3", "P4", "P5"]
SUPERVISED_MASK_LOSS_WEIGHT = 0.1

# -------------------------------------------Tricks config
USE_CONCAT = True
CONCAT_CHANNEL = 1024  # 256
# ROTATE_NMS_USE_GPU = True  # When Train, use GPU NMS, When Test, Use CPU NMS.

ADD_GLOBAL_CTX = True
ADD_EXTR_CONVS_FOR_REG = 8  # use 0 to do not use any extra convs



