"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import math
import numpy as np
import os
import torch

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # Path to pretrained imagenet model
    IMAGENET_MODEL_PATH = os.path.join(os.getcwd(), "resnet50_imagenet.pth")

    # NUMBER OF GPUs to use. For CPU use 0
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # The number of input channels
    NUM_INPUT_CHANNELS = 3
    
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    # MEAN_PIXEL = np.array([0.485, 0.456, 0.406])
    MEAN_PIXEL_TENSOR = torch.from_numpy(MEAN_PIXEL.astype(np.float32)).cuda()

    STD_PIXEL = np.array([1.0, 1.0, 1.0])
    # STD_PIXEL = np.array([0.229, 0.224, 0.225])
    STD_PIXEL_TENSOR = torch.from_numpy(STD_PIXEL.astype(np.float32)).cuda()

    SCALE_PIXEL = 1.0

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.

    # Percent of positive ROIs used to train classifier/mask heads
    if True:
        ## We used a 16G GPU so that we can set TRAIN_ROIS_PER_IMAGE to be 512
        #TRAIN_ROIS_PER_IMAGE = 512
        TRAIN_ROIS_PER_IMAGE = 200
        ROI_POSITIVE_RATIO = 0.33
    else:
        TRAIN_ROIS_PER_IMAGE = 512    
        ROI_POSITIVE_RATIO = 0.5
        pass

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    PARAM_POOL_SIZE = 28
    MASK_SHAPE = [28, 28]
    SUPPORT_SHAPE = [2, 2]
    FINAL_MASK_SHAPE = [224, 224]  # (height, width) of the mini-mask
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    NUM_PARAMETERS = 3

    METADATA = np.array([554.2560, 579.4110, 320, 240, 640, 480, 0, 0, 0, 0])

    IMAGE_MAX_DIM = 640
    IMAGE_MIN_DIM = 480

    GLOBAL_MASK = False

    def __init__(self, options, use_gpu=True):
        """Set values of computed attributes."""
        # Effective batch size
        if self.GPU_COUNT > 0:
            self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        else:
            self.BATCH_SIZE = self.IMAGES_PER_GPU

        # Adjust step size based on batch size
        self.STEPS_PER_EPOCH = self.BATCH_SIZE * self.STEPS_PER_EPOCH

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])
        
        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

        self.MAXDISP = 256
        # self.BASELINE = 0.2
        self.UVD_CONST = 320

        self.BILINEAR_UPSAMPLING = 'bilinear' in options.suffix
        self.FITTING_TYPE = 0

        self.ANCHOR_TYPE = options.anchorType

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        return


class PlaneConfig(Config):
    """Configuration for training on ScanNet.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "plane"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 2  # COCO has 80 classes
    GLOBAL_MASK = False

    EVALUATION_BINS = 6

