"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
cv2.setNumThreads(0)

import math
import os
import re
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torchmetrics
import torchvision
# import torchgeometry
import pytorch_lightning as pl
from pytorch_lightning.profiler import PassThroughProfiler

from scipy import stats

import utils
import utils_cpp_py


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = torch.ones(1, dtype=torch.bool, device=tensor.device)
    unique_bool = torch.cat([first_element, unique_bool], dim=0)
    return tensor[unique_bool]


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


############################################################
#  FPN Graph
############################################################

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels, bilinear_upsampling=False):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.bilinear_upsampling = bilinear_upsampling
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)

        if self.bilinear_upsampling:
            p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2, mode='bilinear')
            p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2, mode='bilinear')
            p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2, mode='bilinear')
        else:
            p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
            p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
            p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
            pass

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        ## P6 is used for the 5th anchor scale in RPN. Generated by
        ## subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out]


############################################################
#  Resnet Graph
############################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False, numInputChannels=3):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
                nn.Conv2d(numInputChannels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
                nn.ReLU(inplace=True),
                SamePad2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    ## Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    ## Apply deltas
    center_y = center_y + deltas[:, 0] * height
    center_x = center_x + deltas[:, 1] * width
    height = height * torch.exp(deltas[:, 2])
    width = width * torch.exp(deltas[:, 3])
    ## Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result


def apply_box_deltas_batch(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [B, N, 4] where each row is y1, x1, y2, x2
    deltas: [B, N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    ## Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    ## Apply deltas
    center_y = center_y + deltas[:, :, 0] * height
    center_x = center_x + deltas[:, :, 1] * width
    height = height * torch.exp(deltas[:, :, 2])
    width = width * torch.exp(deltas[:, :, 3])
    ## Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=2)
    return result


def clip_boxes(boxes, window):
    """
    boxes: [B, N, 4] each col is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    boxes = torch.stack([boxes[:, :, 0].clamp(float(window[0]), float(window[2])),
                         boxes[:, :, 1].clamp(float(window[1]), float(window[3])),
                         boxes[:, :, 2].clamp(float(window[0]), float(window[2])),
                         boxes[:, :, 3].clamp(float(window[1]), float(window[3]))], dim=2)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    ## Box Scores. Use the foreground class confidence. [Batch, num_rois]
    scores = inputs[0][:, :, 1]
    ## Box deltas [batch, num_rois, 4]
    deltas = inputs[1]

    dev = scores.device
    dtype = scores.dtype
    tensor_type = scores.type()
    bsize = scores.shape[0]

    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).type(tensor_type)
    deltas = deltas * std_dev
    ## Improve performance by trimming to top anchors by score
    ## and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True, dim=1)
    order = order[:, :pre_nms_limit]
    scores = scores[:, :pre_nms_limit]
    deltas_limit = torch.zeros((bsize, pre_nms_limit, 4), device=dev, dtype=dtype)
    anchors_limit = torch.zeros((bsize, pre_nms_limit, 4), device=dev, dtype=dtype)
    for b in range(bsize):
        deltas_limit[b, :, :] = deltas[b, order[b], :]
        anchors_limit[b, :, :] = anchors[order[b], :]

    ## Apply deltas to anchors to get refined anchors.
    ## [batch, N, (y1, x1, y2, x2)]
    boxes = apply_box_deltas_batch(anchors_limit, deltas_limit)

    ## Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    ## Filter out small boxes
    ## According to Xinlei Chen's paper, this reduces detection accuracy
    ## for small objects, so we're skipping it.

    nms_boxes = torch.zeros((bsize, proposal_count, 4), device=dev, dtype=dtype)
    for b in range(bsize):
        ## Non-max suppression
        # keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
        keep = torchvision.ops.nms(boxes[b, :, :].index_select(1, torch.tensor([1, 0, 3, 2], device=dev, dtype=torch.long)),
                                   scores[b, :],
                                   nms_threshold)

        keep = keep[:proposal_count]
        nms_boxes[b, :keep.shape[0], :] = boxes[b, keep, :]

    ## Normalize dimensions to range of 0 to 1.
    norm = torch.tensor([height, width, height, width], device=dev, dtype=dtype)
    normalized_boxes = nms_boxes / norm

    return normalized_boxes


############################################################
#  ROIAlign Layer
############################################################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    bsize = boxes.shape[0]
    nsize = boxes.shape[1]
    chsize = feature_maps[0].shape[1]

    dev = boxes.device
    dtype = boxes.dtype
    tensor_type = boxes.type()

    ## Stop gradient propogation to ROI proposals
    boxes = boxes.detach()

    ## Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=2)
    h = y2 - y1
    w = x2 - x1

    ## Equation 1 in the Feature Pyramid Networks paper. Account for
    ## the fact that our coordinates are normalized here.
    ## e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = image_shape[0]*image_shape[1]

    # roi_level = 4 + log2(torch.sqrt(h*w)/(640.0/torch.sqrt(image_area)))
    roi_level = 4 + torch.log2(torch.sqrt(h*w)/(224.0/math.sqrt(image_area)))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2, 5)

    ## Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        # remove last dummy dimension
        ix = (roi_level == level)[:, :, 0]
        if not ix.any():
            continue
        ix = torch.nonzero(ix)
        level_boxes = boxes[ix[:, 0], ix[:, 1], :]
        batch_ix = ix[:, 0:1]

        ## Keep track of which box is mapped to which level
        box_to_level.append(ix)

        ## Stop gradient propogation to ROI proposals
        # level_boxes = level_boxes.detach()
        # batch_ix = batch_ix.detach()

        ## Crop and Resize
        ## From Mask R-CNN paper: "We sample four regular locations, so
        ## that we can evaluate either max or average pooling. In fact,
        ## interpolating only a single value at each bin center (without
        ## pooling) is nearly as effective."
        #
        ## Here we use the simplified approach of a single value per bin,
        ## which is how it's done in tf.crop_and_resize()
        ## Result: [num_boxes in all batches, channels, pool_height, pool_width, channels]

        pooled_features = utils.roi_align_batch(feature_maps[i],
                                          torch.cat([batch_ix.float(), level_boxes], dim=1),
                                          (pool_size, pool_size))
        pooled.append(pooled_features)

    ## Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)
    ## Pack box_to_level mapping into one array and add another
    ## column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    ## Rearrange pooled features to match the order of the original boxes
    # Sort according to batch and then
    # _, box_to_level_ix = torch.sort(box_to_level[:, 0] * nsize + box_to_level[:, 1])
    # pooled_ord = pooled[box_to_level_ix, :, :, :]
    # pooled_ord = pooled_ord.view(bsize, nsize, chsize, pool_size, pool_size)

    pooled_ord = torch.zeros((bsize, nsize, chsize, pool_size, pool_size), device=dev, dtype=dtype)
    pooled_ord[box_to_level[:, 0], box_to_level[:, 1], :, :, :] = pooled

    return pooled_ord


def flat_roi_align(inputs, pool_size, image_shape, sampling_ratio=-1):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [batch, num_boxes, channels, height, width].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]
    cooridnates = inputs[1]

    bsize = boxes.shape[0]
    chsize = cooridnates.shape[1]
    num_boxes = boxes.shape[1]
    device = boxes.device

    boxes = boxes.view(-1, 4)
    batch_ix = torch.arange(bsize, device=device).repeat_interleave(num_boxes).view(-1, 1)

    ## Stop gradient propogation to ROI proposals
    boxes = boxes.detach()

    pooled_features = utils.roi_align_batch(cooridnates,
                                      torch.cat([batch_ix.float(), boxes], dim=1),
                                      (pool_size, pool_size),
                                      sampling_ratio=sampling_ratio)

    pooled_features = pooled_features.view(bsize, num_boxes, chsize, pool_size, pool_size)

    return pooled_features


def masked_roi_align(config, inputs, masks, pool_size, image_shape, sampling_ratio=-1):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_size: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [batch, num_boxes, channels, height, width].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]
    feat = inputs[1]

    bsize = boxes.shape[0]
    chsize = feat.shape[1]
    hsize = feat.shape[2]
    wsize = feat.shape[3]
    nsize = boxes.shape[1]
    device = boxes.device

    # boxes = boxes.view(-1, 4)
    # batch_ix = torch.arange(bsize, device=device).repeat_interleave(num_boxes).view(-1, 1)

    ## Stop gradient propogation to ROI proposals
    boxes = boxes.detach()
    masks = masks.detach()

    # masks = torch.where(masks > 0.5, 1.0, 0.0)

    pooled_features = torch.zeros((bsize, nsize, chsize, pool_size, pool_size), device=device)
    for b in range(bsize):
        for n in range(nsize):
            if int((boxes[b, n, 2] - boxes[b, n, 0]) * hsize) * int((boxes[b, n, 3] - boxes[b, n, 1]) * wsize) > 0.0:
                full_mask = utils.resize_mask_relative(hsize, wsize, boxes[b, n, :], masks[b, n, :, :]).unsqueeze(0)
                full_mask = torch.where(full_mask > 0.5, 1.0, 0.0)

                cur_feat = feat[b, :, :, :].clone()
                cur_feat[torch.logical_not(full_mask > 0.5)] = 0.0

                cur_pooled_feat = utils.roi_align(cur_feat.unsqueeze(0),
                                                  [boxes[b, n:n+1, :]],
                                                  pool_size,
                                                  sampling_ratio)

                cur_pooled_mask = utils.roi_align(full_mask.unsqueeze(0),
                                                  [boxes[b, n:n + 1, :]],
                                                  pool_size,
                                                  sampling_ratio)

                cur_pooled_feat[cur_pooled_mask < 0.5] = 0.0
                # When pooling features were averaged, i.e. divided by the number of bins.
                # To correct for empty bins we have to divide by fraction of masked area.
                cur_pooled_feat /= cur_pooled_mask.clamp(min=1.0e-3)

                pooled_features[b, n, :, :, :] = cur_pooled_feat

    return pooled_features


############################################################
##  Detection Target Layer
############################################################

def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    ## 1. Tile boxes2 and repeate boxes1. This allows us to compare
    ## every boxes1 against every boxes2 without loops.
    ## TF doesn't have an equivalent to np.repeate() so simulate it
    ## using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    ## 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = torch.zeros(y1.size()[0], device=boxes1.device)

    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    ## 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    ## 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def detection_target_layer(proposals, gt_class_ids, gt_plane_ids, gt_boxes, gt_masks, gt_parameters, config, camera):
    """Subsamples proposals and generates target box refinment, class_ids,
    and cur_masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, MAX_GT_INSTANCES, height, width] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and cur_masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    bsize = proposals.shape[0]
    device = proposals.device

    rois = torch.zeros((bsize, config.TRAIN_ROIS_PER_IMAGE, 4), device=device)
    roi_gt_class_ids = torch.zeros((bsize, config.TRAIN_ROIS_PER_IMAGE), device=device)
    roi_gt_plane_ids = torch.zeros((bsize, config.TRAIN_ROIS_PER_IMAGE), device=device)
    deltas = torch.zeros((bsize, config.TRAIN_ROIS_PER_IMAGE, 4), device=device)
    masks = torch.zeros((bsize, config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1]), device=device)
    roi_gt_parameters = torch.zeros((bsize, config.TRAIN_ROIS_PER_IMAGE, config.NUM_PARAMETERS), device=device)
    plane_params = torch.zeros((bsize, config.TRAIN_ROIS_PER_IMAGE, 3), device=device)

    for b in range(bsize):
        # Removing dummy dimension
        valid_idxs = torch.nonzero(gt_class_ids[b, :] > 0)[:, 0]

        if len(valid_idxs) > 0:
            cur_proposals = proposals[b, :, :]
            cur_gt_class_ids = gt_class_ids[b, valid_idxs]
            cur_gt_plane_ids = gt_plane_ids[b, valid_idxs]
            cur_gt_boxes = gt_boxes[b, valid_idxs, :]
            cur_gt_masks = gt_masks[b, valid_idxs, :, :]
            cur_gt_parameters = gt_parameters[b, valid_idxs, :]

            ## Compute overlaps matrix [proposals, gt_boxes]
            overlaps = bbox_overlaps(cur_proposals, cur_gt_boxes)

            ## Determine postive and negative ROIs
            roi_iou_max = torch.max(overlaps, dim=1)[0]

            ## 1. Positive ROIs are those with >= 0.5 IoU with a GT box
            positive_roi_bool = roi_iou_max >= 0.5
            #print('positive count', positive_roi_bool.sum())

            ## Subsample ROIs. Aim for 33% positive
            ## Positive ROIs
            if positive_roi_bool.sum() > 0:
                positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

                positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                                     config.ROI_POSITIVE_RATIO)
                rand_idx = torch.randperm(positive_indices.size()[0], device=device)
                rand_idx = rand_idx[:positive_count]

                positive_indices = positive_indices[rand_idx]
                positive_count = positive_indices.size()[0]
                positive_rois = cur_proposals[positive_indices, :]

                ## Assign positive ROIs to GT boxes.
                positive_overlaps = overlaps[positive_indices, :]
                roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
                cur_roi_gt_boxes = cur_gt_boxes[roi_gt_box_assignment, :]
                cur_roi_gt_class_ids = cur_gt_class_ids[roi_gt_box_assignment]
                cur_roi_gt_plane_ids = cur_gt_plane_ids[roi_gt_box_assignment]
                cur_roi_gt_parameters = cur_gt_parameters[roi_gt_box_assignment]

                ## Compute bbox refinement for positive ROIs
                cur_deltas = utils.box_refinement(positive_rois.detach(), cur_roi_gt_boxes.detach())
                std_dev = torch.from_numpy(config.BBOX_STD_DEV).float().to(device)
                cur_deltas /= std_dev

                ## Assign positive ROIs to GT cur_masks
                roi_masks = cur_gt_masks[roi_gt_box_assignment]

                ## Compute mask targets
                # y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
                # y1 /= disp.shape[2]
                # x1 /= disp.shape[3]
                # y2 /= disp.shape[2]
                # x2 /= disp.shape[3]
                # boxes_disp = torch.cat([y1, x1, y2, x2], dim=1)

                boxes = positive_rois
                if config.USE_MINI_MASK:
                    ## Transform ROI corrdinates from normalized image space
                    ## to normalized mini-mask space.
                    y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
                    gt_y1, gt_x1, gt_y2, gt_x2 = cur_roi_gt_boxes.chunk(4, dim=1)
                    gt_h = gt_y2 - gt_y1
                    gt_w = gt_x2 - gt_x1
                    y1 = (y1 - gt_y1) / gt_h
                    x1 = (x1 - gt_x1) / gt_w
                    y2 = (y2 - gt_y1) / gt_h
                    x2 = (x2 - gt_x1) / gt_w
                    boxes = torch.cat([y1, x1, y2, x2], dim=1)

                ## Threshold mask pixels at 0.5 to have GT cur_masks be 0 or 1 to use with
                ## binary cross entropy loss.
                # Add channel dimension
                cur_masks = utils.roi_align(roi_masks.unsqueeze(1),
                                   boxes.detach().chunk(boxes.shape[0], dim=0),
                                   (config.MASK_SHAPE[0], config.MASK_SHAPE[1]))
                # Remove channel dimension
                cur_masks = torch.round(cur_masks).squeeze(1)

                fx = camera[b, 0]
                fy = camera[b, 1]
                cx = camera[b, 2]
                cy = camera[b, 3]
                dcy = (camera[b, 4] - camera[b, 5]) / 2
                w = camera[b, 4]
                h = camera[b, 5]
                # b = config.BASELINE
                a = config.UVD_CONST

                roi_gt_offsets = cur_roi_gt_parameters.norm(dim=1, keepdim=True)
                roi_gt_normals = cur_roi_gt_parameters / roi_gt_offsets
                # add and remove batch dimension
                roi_gt_normals_uvd = utils.normal_to_normal_uvd(camera[b:b + 1, :],
                                                                config,
                                                                roi_gt_normals.unsqueeze(0),
                                                                320, 320).squeeze(0)
                roi_gt_normals_uvd = F.normalize(roi_gt_normals_uvd, dim=1)

                cur_plane_params = roi_gt_normals_uvd

                roi_gt_class_ids[b, 0:positive_count] = cur_roi_gt_class_ids
                roi_gt_plane_ids[b, 0:positive_count] = cur_roi_gt_plane_ids
                deltas[b, 0:positive_count, :] = cur_deltas
                masks[b, 0:positive_count, :, :] = cur_masks
                roi_gt_parameters[b, 0:positive_count, :] = cur_roi_gt_parameters
                plane_params[b, 0:positive_count, :] = cur_plane_params
            else:
                positive_count = 0
                positive_rois = torch.zeros((positive_count, 4), device=device)

            ## 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
            negative_roi_bool = roi_iou_max < 0.5
            # negative_roi_bool = negative_roi_bool & no_crowd_bool
            ## Negative ROIs. Add enough to maintain positive:negative ratio.
            if (negative_roi_bool > 0).sum() > 0 and positive_count > 0:
                negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
                r = 1.0 / config.ROI_POSITIVE_RATIO
                negative_count = int(r * positive_count - positive_count)
                rand_idx = torch.randperm(negative_indices.size()[0], device=device)
                rand_idx = rand_idx[:negative_count]
                negative_indices = negative_indices[rand_idx]
                negative_count = negative_indices.size()[0]
                negative_rois = cur_proposals[negative_indices, :]
            else:
                negative_count = 0
                negative_rois = torch.zeros((negative_count, 4), device=device)

            #print('count', positive_count, negative_count)
            #print(cur_roi_gt_class_ids)

            rois[b, 0:(positive_count + negative_count), :] = torch.cat([positive_rois, negative_rois], dim=0)

    return rois, roi_gt_class_ids, deltas, masks, plane_params, roi_gt_plane_ids


############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes = torch.stack([boxes[:, 0].clamp(float(window[0]), float(window[2])), boxes[:, 1].clamp(float(window[1]), float(window[3])), boxes[:, 2].clamp(float(window[0]), float(window[2])), boxes[:, 3].clamp(float(window[1]), float(window[3]))], dim=-1)
    return boxes


def refine_detections(rois,
                      probs,
                      deltas,
                      plane_id_desc,
                      window,
                      config, return_indices=False, use_nms=1, one_hot=True):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score, param_1, ..., param_N)]
    """

    dev = rois.device
    dtype = rois.dtype
    tensor_type = rois.type()

    ## Class IDs per ROI

    if len(probs.shape) == 1:
        class_ids = probs.long()
    else:
        _, class_ids = torch.max(probs, dim=1)
        pass

    ## Class probability of the top class of each ROI
    ## Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0], device=dev).long()

    if len(probs.shape) == 1:
        class_scores = torch.ones(class_ids.shape, device=dev)
        deltas_specific = deltas
    else:
        class_scores = probs[idx, class_ids.data]
        deltas_specific = deltas[idx, class_ids.data]

    class_parameters = torch.zeros((class_ids.shape[0], config.NUM_PARAMETERS), device=dev)

    ## Apply bounding box deltas
    ## Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).type(tensor_type)

    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    ## Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = torch.tensor([height, width, height, width], device=dev, dtype=dtype)

    refined_rois = refined_rois * scale
    ## Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    ## Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    ## TODO: Filter out boxes with zero area

    ## Filter out background boxes
    keep_bool = class_ids > 0

    ## Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE and False:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)

    keep_bool = keep_bool & (refined_rois[:, 2] > refined_rois[:, 0]) & (refined_rois[:, 3] > refined_rois[:, 1])

    if keep_bool.sum() == 0:
        if return_indices:
            return [torch.zeros((0, 4 + 2 + config.NUM_PARAMETERS + config.DESC_LEN), device=dev, dtype=dtype),
                    torch.zeros(0, device=dev, dtype=torch.long),
                    torch.zeros((0, 4), device=dev, dtype=dtype)]
        else:
            return torch.zeros((0,  4 + 2 + config.NUM_PARAMETERS + config.DESC_LEN), device=dev, dtype=dtype)
        pass

    keep = torch.nonzero(keep_bool)[:,0]

    if use_nms == 2:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        ixs = torch.arange(len(pre_nms_class_ids), device=dev, dtype=torch.long)
        ## Sort
        ix_rois = pre_nms_rois
        ix_scores = pre_nms_scores
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data,:]

        # nms_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
        nms_keep = torchvision.ops.nms(ix_rois.index_select(1, torch.tensor([1, 0, 3, 2], device=dev, dtype=torch.long)),
                                       ix_scores,
                                       config.DETECTION_NMS_THRESHOLD)
        nms_keep = keep[ixs[order[nms_keep].data].data]
        keep = intersect1d(keep, nms_keep)
    elif use_nms == 1:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
            ## Pick detections of this class
            ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

            ## Sort
            ix_rois = pre_nms_rois[ixs.data]
            ix_scores = pre_nms_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order.data, :]

            # class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
            class_keep = torchvision.ops.nms(ix_rois.index_select(1, torch.tensor([1, 0, 3, 2], device=dev, dtype=torch.long)),
                                             ix_scores,
                                             config.DETECTION_NMS_THRESHOLD)

            ## Map indicies
            class_keep = keep[ixs[order[class_keep].data].data]

            if i==0:
                nms_keep = class_keep
            else:
                nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
        keep = intersect1d(keep, nms_keep)
    else:
        pass

    ## Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]
    #print('num detectinos', len(keep))

    ### Apply plane anchors
    # class_parameters = config.applyAnchorsTensor(class_ids, class_parameters)
    ## Arrange output as [N, (y1, x1, y2, x2, class_id, score, parameters, desc)]
    ## Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1),
                        class_parameters[keep.data],
                        plane_id_desc[keep.data]), dim=1)

    if return_indices:
        ori_rois = rois * scale
        ori_rois = clip_to_window(window, ori_rois)
        ori_rois = torch.round(ori_rois)
        ori_rois = ori_rois[keep.data]
        return result, keep.data, ori_rois

    return result


def detection_layer(config,
                    rois,
                    mrcnn_class,
                    mrcnn_bbox,
                    mrcnn_plane_id_desc,
                    image_meta, use_nms=1, one_hot=True):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, score, param_1, ..., param_N))] in pixels
    """

    _, _, window, _ = utils.parse_image_meta_torch(image_meta)

    bsize = rois.shape[0]
    device = rois.device

    detections = torch.zeros((bsize,
                              config.DETECTION_MAX_INSTANCES,
                              4 + 2 + config.NUM_PARAMETERS + config.DESC_LEN),
                             device=device)

    for b in range(bsize):
        cur_detections = refine_detections(rois[b, :, :],
                                           mrcnn_class[b, :, :],
                                           mrcnn_bbox[b, :, :, :],
                                           mrcnn_plane_id_desc[b, :, :],
                                           window[b, :],
                                           config,
                                           use_nms=use_nms, one_hot=one_hot)
        num_detections = cur_detections.shape[0]
        detections[b, 0:num_detections, :] = cur_detections

    return detections


############################################################
#  Region Proposal Network
############################################################

class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    """

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        ## Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        ## Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        ## Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0,2,3,1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        ## Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        ## Bounding box refinement. [batch, H, W, anchors per location, depth]
        ## where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        ## Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0,2,3,1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes, num_plane_ids, desc_len):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_plane_ids = num_plane_ids
        self.desc_len = desc_len
        self.padding = SamePad2d(kernel_size=3, stride=1)
        # self.conv1 = nn.Conv2d(self.depth + 64, self.depth + 64, kernel_size=3, stride=1)
        # self.bn1 = nn.BatchNorm2d(self.depth + 64, eps=0.001, momentum=0.01)
        # self.conv1b = nn.Conv2d(self.depth + 64, 1024, kernel_size=self.pool_size, stride=1)
        # self.bn1b = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv1 = nn.Conv2d(self.depth + 64, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

        self.linear_plane_id = nn.Linear(1024, self.desc_len)
        self.dropout_plane_id = nn.Dropout(p=0.5)
        self.linear_plane_id_2 = nn.Linear(self.desc_len, self.num_plane_ids)

    def forward(self, x, rois, ranges, pool_features=True):
        bsize = rois.shape[0]
        nsize = rois.shape[1]

        # [B, N, C, H, W]
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = x.view(bsize * nsize, self.depth, self.pool_size, self.pool_size)
        # [B, N, 3, H, W]
        ranges = flat_roi_align([rois] + [ranges, ], self.pool_size, self.image_shape)
        ranges = ranges.view(bsize * nsize, 64, self.pool_size, self.pool_size)

        roi_features = torch.cat([x, ranges], dim=1)
        # roi_features = torch.cat([x, ranges], dim=1)
        # x = self.conv1(self.padding(roi_features))
        # x = self.bn1(x)
        # x = self.conv1b(x)
        # x = self.bn1b(x)
        x = self.conv1(roi_features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        mrcnn_plane_id_desc = self.linear_plane_id(x)
        # mrcnn_plane_id_desc = F.normalize(mrcnn_plane_id_desc, dim=1)
        mrcnn_plane_id_desc_dropout = self.dropout_plane_id(mrcnn_plane_id_desc)
        mrcnn_plane_id_logits = self.linear_plane_id_2(mrcnn_plane_id_desc_dropout)
        mrcnn_plane_id_prob = self.softmax(mrcnn_plane_id_logits)

        mrcnn_class_logits = mrcnn_class_logits.view(bsize, nsize, self.num_classes)
        mrcnn_probs = mrcnn_probs.view(bsize, nsize, self.num_classes)
        mrcnn_bbox = mrcnn_bbox.view(bsize, nsize, self.num_classes, 4)
        mrcnn_plane_id_desc = mrcnn_plane_id_desc.view(bsize, nsize, self.desc_len)
        mrcnn_plane_id_logits = mrcnn_plane_id_logits.view(bsize, nsize, self.num_plane_ids)
        # mrcnn_plane_id_prob = mrcnn_plane_id_prob.view(bsize, nsize, self.num_plane_ids)
        roi_features = roi_features.view(bsize, nsize, self.depth + 64, self.pool_size, self.pool_size)

        if pool_features:
            return [mrcnn_class_logits,
                    mrcnn_probs,
                    mrcnn_bbox,
                    mrcnn_plane_id_desc,
                    mrcnn_plane_id_logits,
                    roi_features]
        else:
            return [mrcnn_class_logits,
                    mrcnn_probs,
                    mrcnn_bbox,
                    mrcnn_plane_id_desc,
                    mrcnn_plane_id_logits]


class Mask(nn.Module):
    def __init__(self, config, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.config = config
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois, pool_features=True):
        bsize = rois.shape[0]
        nsize = rois.shape[1]

        if pool_features:
            roi_features = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        else:
            roi_features = x
        roi_features = roi_features.view(bsize * nsize, self.depth, self.pool_size, self.pool_size)
        x = self.conv1(self.padding(roi_features))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)

        x_mask = self.deconv(x)
        x_mask = self.relu(x_mask)
        x_mask = self.conv5(x_mask)

        x_mask = self.sigmoid(x_mask)

        x_mask = x_mask.view(bsize, nsize, self.num_classes, 2 * self.pool_size, 2 * self.pool_size)
        roi_features = roi_features.view(bsize, nsize, self.depth, self.pool_size, self.pool_size)

        return x_mask, roi_features


class PlaneParams(nn.Module):
    def __init__(self, config, depth, pool_size, image_shape, num_classes):
        super(PlaneParams, self).__init__()
        self.config = config
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.num_feats = 128
        self.num_coord_feats = 64
        self.num_pts = 1
        self.padding2d = SamePad2d(kernel_size=3, stride=1)

        self.relu = nn.ReLU(inplace=True)

        # self.linear1 = nn.Linear(self.num_feats, self.num_feats)
        #
        # self.linear2 = nn.Linear(self.num_feats, 3)

        self.conv1 = nn.Conv2d(self.num_feats,
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv2 = nn.Conv2d(128,
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        self.conv_agg = nn.Conv2d(128,
                                  1024,
                                  kernel_size=self.pool_size // 4)

        self.linear = nn.Linear(1024, 3)

    def forward(self, normal_feat, rois, mask):
        bsize = rois.shape[0]
        nsize = rois.shape[1]

        # normal_feat = masked_roi_align(self.config, [rois] + [normal_feat], mask, self.pool_size, self.image_shape)
        # normal_feat = normal_feat.view(bsize * nsize,
        #                                self.num_feats)

        # x = self.linear1(normal_feat)
        # x = self.relu(x)
        # vals = self.linear2(x)
        # vals = vals.view(bsize, nsize, 3)


        normal_feat = flat_roi_align([rois] + [normal_feat], self.pool_size, self.image_shape)
        normal_feat = normal_feat.view(bsize * nsize,
                                       self.num_feats,
                                       self.pool_size,
                                       self.pool_size)
        # zero features for non planar region
        mask = mask.view(bsize * nsize,
                         self.pool_size,
                         self.pool_size)
        nonmask_mask = torch.nonzero(mask[:, :, :] <= 0.5)
        normal_feat[nonmask_mask[:, 0], :, nonmask_mask[:, 1], nonmask_mask[:, 2]] = 0

        x = self.conv1(normal_feat)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv_agg(x)
        x = x.view(bsize * nsize, 1024)
        x = self.relu(x)

        vals = self.linear(x)
        vals = vals.view(bsize, nsize, 3)

        return vals


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()

        self.register_buffer('disp', torch.arange(0, maxdisp).view(1, maxdisp, 1, 1))

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x*disp, dim=1, keepdim=True)

        stddev = torch.sum(x * (disp - out).square(), dim=1, keepdim=True).sqrt()

        return out, stddev


class Regression2D(nn.Module):
    def __init__(self, maxval):
        super(Regression2D, self).__init__()

        vals = torch.arange(0, maxval)
        self.register_buffer('vals_x',
                             vals.view(maxval, 1).repeat(1, maxval).view(1, maxval, maxval, 1, 1))
        self.register_buffer('vals_y',
                             vals.view(1, maxval).repeat(maxval, 1).view(1, maxval, maxval, 1, 1))

    def forward(self, probs):
        out_x = torch.sum(probs * self.vals_x, dim=[1, 2])
        out_y = torch.sum(probs * self.vals_y, dim=[1, 2])
        out = torch.stack([out_x, out_y], dim=1)
        return out


def make_conv_bn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes,
                                   out_planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=dilation if dilation > 1 else pad,
                                   dilation=dilation,
                                   bias=True),
                         nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01))


def make_conv3d_bn(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes,
                                   out_planes,
                                   kernel_size=kernel_size,
                                   padding=pad,
                                   stride=stride,
                                   bias=True),
                         nn.BatchNorm3d(out_planes, eps=0.001, momentum=0.01))


def make_conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, activation=True):
    if activation:
        return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2,
                          bias=True),
                nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(make_conv_bn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = make_conv_bn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class FeatureExtraction(nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(make_conv_bn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       make_conv_bn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       make_conv_bn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     make_conv_bn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     make_conv_bn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     make_conv_bn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((4, 4), stride=(4, 4)),
                                     make_conv_bn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(make_conv_bn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=True))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=True),
                    nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear',
                                       align_corners=False)

        output_feature = torch.cat(
                (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class DepthStereo(nn.Module):
    def __init__(self, maxdisp, im_h, im_w, inplanes):
        super(DepthStereo, self).__init__()

        self.maxdisp = maxdisp
        self.im_h = im_h
        self.im_w = im_w
        self.inplanes = inplanes

        self.feature_extraction = FeatureExtraction()

        self.dres0 = nn.Sequential(make_conv3d_bn(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   make_conv3d_bn(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(make_conv3d_bn(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   make_conv3d_bn(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(make_conv3d_bn(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   make_conv3d_bn(32, 32, 3, 1, 1))

        self.dres3 = nn.Sequential(make_conv3d_bn(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   make_conv3d_bn(32, 32, 3, 1, 1))

        self.dres4 = nn.Sequential(make_conv3d_bn(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   make_conv3d_bn(32, 32, 3, 1, 1))

        self.classify = nn.Sequential(make_conv3d_bn(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=True))

        self.disparity_regression = DisparityRegression(self.maxdisp)

        self.wc0 = nn.Sequential(make_conv3d_bn(67, 128, 3, (2, 1, 1), 1),
                                 nn.ReLU(inplace=True),
                                 make_conv3d_bn(128, 128, 3, (2, 1, 1), 1),
                                 nn.ReLU(inplace=True),
                                 make_conv3d_bn(128, 128, 3, (2, 1, 1), 1),
                                 nn.ReLU(inplace=True),
                                 make_conv3d_bn(128, 128, 3, (2, 1, 1), 1),
                                 nn.ReLU(inplace=True),
                                 make_conv3d_bn(128, 128, 3, (2, 1, 1), 1),
                                 nn.ReLU(inplace=True),
                                 make_conv3d_bn(128, 128, 3, (2, 1, 1), 1))

        self.n_convs = nn.Sequential(
                make_conv(128 + 32, 128, 3, 1, 1),
                make_conv(128, 128, 3, 1, 2),
                make_conv(128, 128, 3, 1, 4),
                make_conv(128, 128, 3, 1, 8),
                make_conv(128, 128, 3, 1, 16),
                make_conv(128, 128, 3, 1, 1),
        )

        self.n_convs_last = make_conv(128, 3, 3, 1, 1, activation=False)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(.5 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(.5 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward_norm(self, config, camera, cost_vol, feat_left):
        fx = camera[:, 0]
        fy = camera[:, 1]
        cx = camera[:, 2]
        cy = camera[:, 3]
        dcy = (camera[:, 4] - camera[:, 5]) / 2
        b = camera[:, 6]
        im_b = cost_vol.shape[0]
        im_d = cost_vol.shape[2]
        im_h = cost_vol.shape[3]
        im_w = cost_vol.shape[4]
        factor = self.im_w / im_w

        coords = utils.get_coords_uvd_pad_torch_batch(camera, config, config.MAXDISP, 5, 4, 320, 320)

        wc = torch.cat([coords, cost_vol], dim=1)
        wc = wc.contiguous()

        normal_feat = self.wc0(wc)
        normal_feat = normal_feat.view(im_b, 128, im_h, im_w)
        normal_feat = torch.cat([normal_feat, feat_left], dim=1)

        normal_feat = self.n_convs(normal_feat)

        normals = self.n_convs_last(normal_feat)

        normals = F.interpolate(normals, [self.im_h, self.im_w], mode='bilinear', align_corners=False)

        normals = F.normalize(normals, dim=1)

        return normals, normal_feat

    def forward(self, config, camera, im_left, im_right):
        dev = im_left.device
        bsize = im_left.shape[0]

        im_left = utils.unmold_image_torch(im_left, config) / 255.0
        im_right = utils.unmold_image_torch(im_right, config) / 255.0

        # mean = torch.tensor([0.485, 0.456, 0.406], device=dev)
        # stddev = torch.tensor([0.229, 0.224, 0.225], device=dev)
        mean = torch.tensor([0.5, 0.5, 0.5], device=dev)
        stddev = torch.tensor([0.5, 0.5, 0.5], device=dev)
        im_left = (im_left - mean[None, :, None, None]) / stddev[None, :, None, None]
        im_right = (im_right - mean[None, :, None, None]) / stddev[None, :, None, None]

        feat_left = self.feature_extraction(im_left)
        feat_right = self.feature_extraction(im_right)

        # matching
        cost = torch.zeros((feat_left.size()[0],
                            feat_left.size()[1] * 2,
                            self.maxdisp // 4,
                            feat_left.size()[2],
                            feat_left.size()[3]), device=dev)

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :feat_left.size()[1], i, :, i:] = feat_left[:, :, :, i:]
                cost[:, feat_left.size()[1]:, i, :, i:] = feat_right[:, :, :, :-i]
            else:
                cost[:, :feat_left.size()[1], i, :, :] = feat_left
                cost[:, feat_left.size()[1]:, i, :, :] = feat_right
        cost = cost.contiguous()

        cost0 = self.dres0(cost)

        cost_in0 = cost0.clone()

        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        out_cost = torch.cat((cost_in0, cost0.clone()), dim=1)

        cost0 = self.classify(cost0)

        cost0 = F.interpolate(cost0, [self.maxdisp, self.im_h, self.im_w], mode='trilinear', align_corners=False)
        cost0 = torch.squeeze(cost0, 1)
        pred0 = F.softmax(cost0, dim=1)

        pred0, pred0_stddev = self.disparity_regression(pred0)

        pred_norm, normal_feat = self.forward_norm(config, camera, out_cost, feat_left)
        # pred_norm = torch.zeros((bsize, 3, self.im_h, self.im_w), device=im_left.device)

        return pred0, pred0, pred0, pred0_stddev, normal_feat, pred_norm


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    ## Positive and Negative anchors contribute to the loss,
    ## but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    if len(indices) > 0:
        ## Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = rpn_class_logits[indices[:, 0], indices[:, 1], :]
        anchor_class = anchor_class[indices[:, 0], indices[:, 1]]

        ## Crossentropy loss
        loss = F.cross_entropy(rpn_class_logits, anchor_class)
    else:
        loss = torch.tensor(0, device=rpn_match.device)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Positive anchors contribute to the loss, but negative and
    ## neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match == 1)

    if len(indices) > 0:
        ## Pick bbox deltas that contribute to the loss
        rpn_bbox = rpn_bbox[indices[:, 0], indices[:, 1]]

        ## Trim target bounding box deltas to the same length as rpn_bbox.
        target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]

        ## Smooth L1 loss
        loss = F.smooth_l1_loss(rpn_bbox, target_bbox)
    else:
        loss = torch.tensor(0, device=rpn_match.device)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    bsize = target_class_ids.shape[0]
    nsize = target_class_ids.shape[1]

    ## Loss
    if len(target_class_ids) > 0:
        loss = F.cross_entropy(pred_class_logits.view(bsize * nsize, -1),
                               target_class_ids.long().view(bsize * nsize))
    else:
        loss = torch.tensor(0, device=target_class_ids.device)

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    bsize = target_class_ids.shape[0]
    nsize = target_class_ids.shape[1]

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)
        positive_roi_class_ids = target_class_ids[positive_roi_ix[:, 0], positive_roi_ix[:, 1]].long()
        indices = torch.cat((positive_roi_ix, positive_roi_class_ids.view(-1, 1)), dim=1)

        ## Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:, 0], indices[:, 1], :]
        pred_bbox = pred_bbox[indices[:, 0], indices[:, 1], indices[:, 2], :]

        ## Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox,
                                target_bbox)
    else:
        loss = torch.tensor(0, device=target_class_ids.device)

    return loss


def compute_mrcnn_mask_loss(config, target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, num_classes, height, width] float32 tensor
                with values from 0 to 1.
    """
    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)
        positive_roi_class_ids = target_class_ids[positive_roi_ix[:, 0], positive_roi_ix[:, 1]].long()
        indices = torch.cat((positive_roi_ix, positive_roi_class_ids.view(-1, 1)), dim=1)

        ## Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0], indices[:, 1], :, :]

        if config.GLOBAL_MASK:
            y_pred = pred_masks[indices[:, 0], indices[:, 1], 0, :, :]
        else:
            y_pred = pred_masks[indices[:, 0], indices[:, 1], indices[:, 2], :, :]
            pass

        ## Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
        pass
    else:
        loss = torch.tensor(0, device=target_class_ids.device)

    return loss


def compute_mrcnn_parameter_loss(target_parameters, target_class_ids, pred_parameters):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)
        positive_roi_class_ids = target_class_ids[positive_roi_ix[:, 0], positive_roi_ix[:, 1]].long()
        indices = torch.cat((positive_roi_ix, positive_roi_class_ids.view(-1, 1)), dim=1)

        ## Gather the deltas (predicted and true) that contribute to loss
        target_parameters = target_parameters[indices[:, 0], indices[:, 1], :]
        pred_parameters = pred_parameters[indices[:, 0], indices[:, 1], indices[:, 2], :]
        ## Smooth L1 loss
        loss = F.smooth_l1_loss(pred_parameters, target_parameters)
    else:
        loss = torch.tensor(0, device=target_class_ids.device)

    return loss


def compute_mrcnn_plane_params_loss(config, camera, target_plane_params, mrcnn_plane_params, target_class_ids):
    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)

        ## Gather the masks (predicted and true) that contribute to loss
        target_plane_params_pos = target_plane_params[positive_roi_ix[:, 0], positive_roi_ix[:, 1], :]
        mrcnn_plane_params_pos = mrcnn_plane_params[positive_roi_ix[:, 0], positive_roi_ix[:, 1], :]

        loss = F.smooth_l1_loss(mrcnn_plane_params_pos, target_plane_params_pos)
    else:
        loss = torch.tensor(0, device=target_class_ids.device)

    return loss


def compute_mrcnn_plane_id_loss(target_plane_ids, mrcnn_plane_id_logits, target_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """
    bsize = target_class_ids.shape[0]
    nsize = target_class_ids.shape[1]

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)

        target_plane_ids_pos = target_plane_ids[positive_roi_ix[:, 0], positive_roi_ix[:, 1]]
        mrcnn_plane_id_logits_pos = mrcnn_plane_id_logits[positive_roi_ix[:, 0], positive_roi_ix[:, 1], :]

        loss = F.cross_entropy(mrcnn_plane_id_logits_pos,
                               target_plane_ids_pos.long())
    else:
        loss = torch.tensor(0, device=target_class_ids.device)

    return loss


def compute_losses(config, camera, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox,
                   target_class_ids, mrcnn_class_logits,
                   target_deltas, mrcnn_bbox,
                   target_mask, mrcnn_mask,
                   target_plane_params, mrcnn_plane_params,
                   target_plane_ids, mrcnn_plane_id_logits):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(config, target_mask, target_class_ids, mrcnn_mask)
    # mrcnn_parameter_loss = 100*compute_mrcnn_parameter_loss(target_parameters, target_class_ids, mrcnn_parameters)
    mrcnn_plane_params_loss = 100*compute_mrcnn_plane_params_loss(config, camera, target_plane_params, mrcnn_plane_params, target_class_ids)
    mrcnn_plane_id_loss = compute_mrcnn_plane_id_loss(target_plane_ids, mrcnn_plane_id_logits, target_class_ids)

    return [rpn_class_loss,
            rpn_bbox_loss,
            mrcnn_class_loss,
            mrcnn_bbox_loss,
            mrcnn_mask_loss,
            mrcnn_plane_params_loss,
            mrcnn_plane_id_loss]


############################################################
#  MaskRCNN Class
############################################################

class HistRmsMetric(torchmetrics.Metric):
    def __init__(self, nbins, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.nbins = nbins
        self.add_state("sum", default=torch.zeros(nbins), dist_reduce_fx="sum")
        self.add_state("area", default=torch.zeros(nbins), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(nbins), dist_reduce_fx="sum")

        self.conf_level = 0.95

    def update(self, sum: torch.Tensor, area: torch.Tensor, count: torch.Tensor):
        self.sum += sum
        self.area += area
        self.count += count

    def compute(self):
        hist_rmse = (self.sum.float() / self.area.clamp(min=1.0)).sqrt()
        total_rmse = (self.sum.float().sum() / self.area.sum().clamp(min=1.0)).sqrt()

        count_np = self.count.cpu().detach().numpy()
        conf_int = torch.zeros((self.nbins, 2), device=hist_rmse.device, dtype=hist_rmse.dtype)
        for bin_idx in range(self.nbins):
            if count_np[bin_idx] > 0:
                c1, c2 = stats.chi2.ppf([(1 - self.conf_level) / 2, 1 - (1 - self.conf_level) / 2], count_np[bin_idx])
                conf_int[bin_idx, 0] = math.sqrt(count_np[bin_idx] / c2) * hist_rmse[bin_idx]
                conf_int[bin_idx, 1] = math.sqrt(count_np[bin_idx] / c1) * hist_rmse[bin_idx]

        return hist_rmse, conf_int, self.area, self.count, total_rmse


class HistMeanMetric(torchmetrics.Metric):
    def __init__(self, nbins, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.nbins = nbins
        self.add_state("sum", default=torch.zeros(nbins), dist_reduce_fx="sum")
        self.add_state("area", default=torch.zeros(nbins), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(nbins), dist_reduce_fx="sum")

        self.conf_level = 0.95

    def update(self, sum: torch.Tensor, area: torch.Tensor, count: torch.Tensor):
        self.sum += sum
        self.area += area
        self.count += count

    def compute(self):
        hist_mean = self.sum.float() / self.area.clamp(min=1.0)
        total_mean = self.sum.float().sum() / self.area.sum().clamp(min=1.0)

        return hist_mean, self.area, self.count, total_mean


class DescRankMetric(torchmetrics.Metric):
    def __init__(self, desc_len, dist_sync_on_step=False):
        super().__init__(compute_on_step=False, dist_sync_on_step=dist_sync_on_step)

        self.desc_len = desc_len
        self.add_state("plane_ids", default=[], dist_reduce_fx="cat")
        self.add_state("descs", default=[], dist_reduce_fx="cat")
        self.add_state("tss", default=[], dist_reduce_fx="cat")
        self.add_state("cat_idxs", default=[], dist_reduce_fx="cat")
        self.add_state("scene_ids", default=[], dist_reduce_fx="cat")

    def update(self, plane_ids: torch.Tensor, descs: torch.Tensor, tss: torch.Tensor, cat_idxs: torch.Tensor, scene_ids):
        self.plane_ids.extend(plane_ids.split(1))
        self.descs.extend(descs.split(1))
        self.tss.extend(tss.split(1))
        self.cat_idxs.extend(cat_idxs.split(1))
        self.scene_ids.extend(scene_ids)

    def compute(self):
        tss_thresh = 200.0

        if len(self.plane_ids) > 0:
            plane_ids = torch.cat(self.plane_ids)
            descs = torch.cat(self.descs)
            tss = torch.cat(self.tss)
            cat_idxs = torch.cat(self.cat_idxs)

            n_cats = int(cat_idxs.max()) + 1

            rank_q0 = torch.zeros(n_cats, dtype=torch.long, device=self.device)
            rank_q1 = torch.zeros(n_cats, dtype=torch.long, device=self.device)
            rank_q2 = torch.zeros(n_cats, dtype=torch.long, device=self.device)
            rank_q3 = torch.zeros(n_cats, dtype=torch.long, device=self.device)
            rank_q4 = torch.zeros(n_cats, dtype=torch.long, device=self.device)
            rank_mean = torch.zeros(n_cats, dtype=torch.long, device=self.device)

            # convert string scene_id to integer scene_id
            scene_id_idx = {}
            for scene_id in self.scene_ids:
                if scene_id not in scene_id_idx:
                    idx = len(scene_id_idx)
                    scene_id_idx[scene_id] = idx
            scene_ids = torch.zeros_like(tss)
            for i, scene_id in enumerate(self.scene_ids):
                scene_ids[i] = scene_id_idx[scene_id]

            ranks = [[] for i in range(n_cats)]
            for n in range(len(descs)):
                desc_diff = descs - descs[n, :].view(1, -1)
                desc_dist = desc_diff.square().sum(dim=1)
                sorted_idxs = torch.argsort(desc_dist)
                sorted_plane_ids = plane_ids[sorted_idxs]
                sorted_tss = tss[sorted_idxs]
                sorted_scene_ids = scene_ids[sorted_idxs]
                match_mask = torch.logical_and(sorted_plane_ids == plane_ids[n], sorted_idxs != n)
                # if timestamp difference larger than threshold or different scene_id
                tss_mask = torch.logical_or((sorted_tss - tss[n]).abs() > tss_thresh,
                                            sorted_scene_ids != scene_ids[n])
                match_mask = torch.logical_and(match_mask, tss_mask)
                # reject neighboring frames and the plane itself
                valid_mask = torch.logical_and(sorted_idxs != n, tss_mask)
                match_idxs = torch.where(match_mask)[0]
                valid_idxs = torch.where(valid_mask)[0]
                if len(match_idxs) > 0:
                    first_match = torch.min(match_idxs)
                    # number of valid entries before and at the position of the first valid match
                    num_valid = (valid_idxs <= first_match).sum()

                    cat_idx = int(cat_idxs[n])
                    ranks[cat_idx].append(num_valid)

            for cat_idx in range(n_cats):
                n = len(ranks[cat_idx])
                if n > 0:
                    ranks[cat_idx], _ = torch.sort(torch.stack(ranks[cat_idx]))
                    rank_q1[cat_idx] = ranks[cat_idx][n // 4]
                    rank_q2[cat_idx] = ranks[cat_idx][n // 2]
                    rank_q3[cat_idx] = ranks[cat_idx][3 * n // 4]
                    iqr = rank_q3[cat_idx] - rank_q1[cat_idx]
                    mask = torch.logical_and(rank_q1[cat_idx] - 1.5 * iqr <= ranks[cat_idx],
                                             ranks[cat_idx] <= rank_q3[cat_idx] + 1.5 * iqr)
                    ranks[cat_idx] = ranks[cat_idx][mask]
                    rank_q0[cat_idx] = ranks[cat_idx][0]
                    rank_q4[cat_idx] = ranks[cat_idx][-1]
                    rank_mean[cat_idx] = ranks[cat_idx].float().mean()

        else:
            rank_q0 = torch.zeros(0, dtype=torch.long, device=self.device)
            rank_q1 = torch.zeros(0, dtype=torch.long, device=self.device)
            rank_q2 = torch.zeros(0, dtype=torch.long, device=self.device)
            rank_q3 = torch.zeros(0, dtype=torch.long, device=self.device)
            rank_q4 = torch.zeros(0, dtype=torch.long, device=self.device)
            rank_mean = torch.zeros(0, dtype=torch.long, device=self.device)

        return rank_q0, rank_q1, rank_q2, rank_q3, rank_q4, rank_mean


class MaskRCNN(pl.LightningModule):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self,
                 config,
                 options,
                 detect=True,
                 annotations_as_detections=False,
                 export_detections=False,
                 evaluate_descriptors=False,
                 profiler=PassThroughProfiler()):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.options = options
        self.image_log_rate = 100
        self.detect = detect
        self.annotations_as_detections = annotations_as_detections
        self.export_detections = export_detections
        self.evaluate_descriptors = evaluate_descriptors
        self.build()
        self.initialize_weights()

        # used during descriptor evaluation to reject matches from neighboring frames
        self.fps = 30

        self.profiler = profiler

    def build(self):
        """Build Mask R-CNN architecture.
        """

        if self.detect:
            ## Image size must be dividable by 2 multiple times
            h, w = self.config.IMAGE_SHAPE[:2]
            if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
                raise Exception("Image size must be dividable by 2 at least 6 times "
                                "to avoid fractions when downscaling and upscaling."
                                "For example, use 256, 320, 384, 448, 512, ... etc. ")

            ## Build the shared convolutional layers.
            ## Bottom-up Layers
            ## Returns a list of the last layers of each stage, 5 in total.
            ## Don't create the thead (stage 5), so we pick the 4th item in the list.
            resnet = ResNet("resnet101", stage5=True, numInputChannels=self.config.NUM_INPUT_CHANNELS)
            C1, C2, C3, C4, C5 = resnet.stages()

            ## Top-down Layers
            ## TODO: add assert to varify feature map sizes match what's in config
            self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=self.config.BILINEAR_UPSAMPLING)

            ## Generate Anchors
            self.register_buffer('anchors',
                                 torch.from_numpy(utils.generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                                                self.config.RPN_ANCHOR_RATIOS,
                                                                                self.config.BACKBONE_SHAPES,
                                                                                self.config.BACKBONE_STRIDES,
                                                                                self.config.RPN_ANCHOR_STRIDE)).float())

            ## RPN
            self.rpn = RPN(len(self.config.RPN_ANCHOR_RATIOS), self.config.RPN_ANCHOR_STRIDE, 256)

            ## Coordinate feature
            self.range_conv = nn.Conv2d(3, 64, kernel_size=1, stride=1)

            ## FPN Classifier
            self.classifier = Classifier(256,
                                         self.config.POOL_SIZE,
                                         self.config.IMAGE_SHAPE,
                                         self.config.NUM_CLASSES,
                                         self.options.num_plane_ids,
                                         self.config.DESC_LEN)

            ## FPN Mask
            self.mask = Mask(self.config, 256, self.config.MASK_POOL_SIZE, self.config.IMAGE_SHAPE, self.config.NUM_CLASSES)

            self.plane_params = PlaneParams(self.config,
                                            256,
                                            4 * self.config.POOL_SIZE,
                                            self.config.IMAGE_SHAPE,
                                            self.config.NUM_CLASSES)

        self.depth = DepthStereo(self.config.MAXDISP,
                                 self.config.IMAGE_SHAPE[0],
                                 self.config.IMAGE_SHAPE[1],
                                 256)

        self.dist_hist = HistRmsMetric(self.config.EVALUATION_BINS)
        self.norm_hist = HistRmsMetric(self.config.EVALUATION_BINS)
        self.target_dist_hist = HistMeanMetric(self.config.EVALUATION_BINS)
        self.desc_ranks = DescRankMetric(self.config.DESC_LEN)

        ## Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # self.bn_exceptions = {'classifier', 'mask', 'plane_params'}
        # self.bn_exceptions = {'depth'}
        self.bn_exceptions = {}
        for (mname, m) in self.named_children():
            if mname not in self.bn_exceptions:
                m.apply(set_bn_fix)

    def configure_optimizers(self):
        trainable_params = []
        if self.options.trainingMode != '':
            ## Specify which layers to train, default is "all"
            layer_regex = {
                ## all layers but the backbone
                "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                ## From a specific Resnet stage and up
                "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
                ## All layers
                "all": ".*",
                "classifier": "(classifier.*)|(mask.*)|(depth.*)",
                "depth": "(depth.*)",
            }
            assert (self.options.trainingMode in layer_regex.keys())
            layers = layer_regex[self.options.trainingMode]
            self.set_trainable(layers)
            trainable_params = [(name, param) for name, param in self.named_parameters() if bool(re.fullmatch(layers, name))]
        else:
            trainable_params = self.named_parameters()

        trainables_wo_bn = [param for name, param in trainable_params if not 'bn' in name]
        trainables_only_bn = [param for name, param in trainable_params if 'bn' in name]

        # optimizer = optim.SGD([
        #     {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        #     {'params': trainables_only_bn}], lr=self.options.LR, momentum=0.9)
        optimizer = optim.Adam([
            {'params': trainables_wo_bn, 'weight_decay': 0.0001},
            {'params': trainables_only_bn}], lr=self.options.LR)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        if isinstance(self.depth, DepthStereo):
            self.depth.initialize_weights()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def load_weights(self, filepaths):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """

        state_dict = {}
        for file in filepaths:
            cur_state_dict = torch.load(file)
            state_dict.update(cur_state_dict)
        try:
            self.load_state_dict(state_dict, strict=False)
            # for key in self.state_dict().keys():
            #     if key not in state_dict.keys():
            #         print('Uninitialized ', key)
        except:
            print('load only base model')
            try:
                state_dict = {k: v for k, v in state_dict.items() if 'classifier.linear_class' not in k
                              and 'classifier.linear_bbox' not in k
                              and 'mask.conv5' not in k}
                state = self.state_dict()
                state.update(state_dict)
                self.load_state_dict(state)
            except:
                print('change input dimension')
                state_dict = {k: v for k, v in state_dict.items()
                              if 'plane_params' not in k
                              # if 'classifier.linear_class' not in k
                              # and 'plane_params' not in k
                              # and 'classifier.linear_bbox' not in k
                              # and 'classifier.linear_parameters' not in k
                              # and 'mask.conv5' not in k
                              # and 'depth' not in k
                              # and 'mask.conv1' not in k
                              # and 'fpn.C1.0' not in k
                              and 'classifier.conv1' not in k
                              # and 'classifier.bn1' not in k
                              # and 'rpn.conv_shared' not in k
                              and 'depth.wc0' not in k
                              and 'depth.pool1' not in k
                              and 'depth.pool2' not in k
                              and 'depth.pool3' not in k
                              and 'depth.convs' not in k
                              and 'depth.n_convs' not in k
                              }
                state = self.state_dict()
                state.update(state_dict)
                self.load_state_dict(state)
                for key in self.state_dict().keys():
                    if key not in state_dict.keys():
                        print('Uninitialized ', key)

    # def on_after_backward(self):
    #     for k, v in self.named_parameters():
    #         if (v.requires_grad) and ("bias" not in k):
    #             if v.grad is None:
    #                 print(k)
    #             else:
    #                 self.logger.experiment.add_histogram(
    #                     tag=k, values=v.grad, global_step=self.trainer.global_step
    #                 )

    def save_as_annotation(self,
                           batch,
                           batch_idx,
                           camera,
                           detections,
                           masks,
                           target_depth,
                           depth_np,
                           disp_stddev,
                           from_target=True,
                           save_depth=False,
                           save_det=False,
                           output_dir='annotation_inferred'):
        bsize = detections.shape[0]
        nsize = detections.shape[1]
        tensor_type = target_depth.type()
        dev = target_depth.device

        boxes = detections[:, :, 0:4].long()
        class_ids = detections[:, :, 4].long()
        batch_descs = detections[:, :, 9:]

        ranges = utils.get_ranges_pad_torch_batch(camera)

        XYZ = ranges * target_depth
        valid_mask = target_depth > 0.2

        for b in range(bsize):
            fx = camera[b, 0]
            baseline = camera[b, 6]

            annotation_dir = os.path.join(self.options.dataFolder,
                                          'scenes',
                                          batch['scene_id'][b],
                                          output_dir)
            if not os.path.exists(annotation_dir):
                os.mkdir(annotation_dir)
            segmentation_left_dir = os.path.join(annotation_dir, 'segmentation_left')
            if not os.path.exists(segmentation_left_dir):
                os.mkdir(segmentation_left_dir)
            segmentation_right_dir = os.path.join(annotation_dir, 'segmentation_right')
            if not os.path.exists(segmentation_right_dir):
                os.mkdir(segmentation_right_dir)

            if save_depth:
                depth_left_dir = os.path.join(annotation_dir, 'depth_left')
                if not os.path.exists(depth_left_dir):
                    os.mkdir(depth_left_dir)
                depth_stddev_left_dir = os.path.join(annotation_dir, 'depth_stddev_left')
                if not os.path.exists(depth_stddev_left_dir):
                    os.mkdir(depth_stddev_left_dir)

            if save_det:
                det_left_dir = os.path.join(annotation_dir, 'det_left')
                if not os.path.exists(det_left_dir):
                    os.mkdir(det_left_dir)
                det_images = utils.draw_instance_images(self.config,
                                                        batch['left']['image'],
                                                        detections[:, :, 0:4],
                                                        masks,
                                                        detections[:, :, 4])

            planes_filename = os.path.join(self.options.dataFolder,
                                           'scenes',
                                           batch['scene_id'][b],
                                           output_dir,
                                           'planes.npy')

            planes = torch.zeros((0, 4), device=dev)
            if os.path.exists(planes_filename):
                planes = torch.from_numpy(np.load(planes_filename)).type(tensor_type)
            next_id = planes.shape[0]

            descs_filename = os.path.join(self.options.dataFolder,
                                           'scenes',
                                           batch['scene_id'][b],
                                           output_dir,
                                           'descs.npy')
            descs = torch.zeros((0, self.config.DESC_LEN), device=dev)
            if os.path.exists(descs_filename):
                descs = torch.from_numpy(np.load(descs_filename)).type(tensor_type)
            if next_id != descs.shape[0]:
                raise 'Number of descriptors not equal to the number of planes'

            T_c_w = batch['left']['extrinsics'][b]
            T_w_c_inv_t = T_c_w.transpose(0, 1)
            full_masks = []
            areas = []
            plane_eqs = []
            plane_eqs_c = []
            cur_descs = []
            for n in range(nsize):
                if class_ids[b, n] > 0:
                    full_mask = utils.resize_mask_full(self.config, boxes[b, n, :], masks[b, n, class_ids[b, n], :, :])
                    full_mask_valid = torch.logical_and(full_mask > 0.5, valid_mask[b, :, :, :].squeeze(0))
                    # gt points belonging to the plane
                    XYZ_plane_gt = XYZ[b, :, full_mask_valid]

                    area = XYZ_plane_gt.shape[1]
                    if area >= 500:
                        if from_target:
                            # inlier_mask_gt, plane_gt = fit_plane_ransac_torch(XYZ_plane_gt.transpose(0, 1),
                            #                                                   plane_diff_threshold=0.02)
                            inlier_mask_gt_np, plane_gt_np = utils_cpp_py.ransac_plane(XYZ_plane_gt.cpu().numpy(),
                                                                                       50,
                                                                                       0.02,
                                                                                       False)
                            inlier_mask_gt = torch.from_numpy(inlier_mask_gt_np).type_as(planes)
                            plane_gt = torch.from_numpy(plane_gt_np).type_as(planes)

                            inliers_ratio = float(inlier_mask_gt.sum()) / XYZ_plane_gt.shape[1]

                            if plane_gt.norm() < 1.0e-5 or inliers_ratio < 0.6:
                                continue
                            plane = plane_gt / plane_gt.norm().square()
                        else:
                            plane = detections[b, n, 6:9]

                        offset = plane.norm(keepdim=True)
                        normal = plane / offset
                        plane_eq = torch.cat([normal, -offset])

                        plane_eq_w = torch.matmul(T_w_c_inv_t, plane_eq.view(4, 1)).view(4)

                        full_masks.append(full_mask > 0.5)
                        areas.append(area)
                        plane_eqs.append(plane_eq_w)
                        plane_eqs_c.append(plane)
                        cur_descs.append(batch_descs[b, n, :])

            # sort in descending order, bigger planes have priority
            order = sorted([(area, idx) for idx, area in enumerate(areas)], key=lambda plane: -plane[0])
            order_ids = [idx for area, idx in order]
            max_val = 2**31 - 1
            segmentation = max_val * torch.ones((1, 1, target_depth.shape[2], target_depth.shape[3]),
                                                device=dev,
                                                dtype=torch.long)
            planes_c = torch.zeros((0, 3), device=dev)
            for idx in order_ids:
                cur_segmentation = max_val * torch.ones((1, 1, target_depth.shape[2], target_depth.shape[3]),
                                                        device=dev,
                                                        dtype=torch.long)
                cur_segmentation[0, 0, full_masks[idx] > 0.5] = next_id
                segmentation = torch.minimum(segmentation, cur_segmentation)
                planes = torch.cat([planes, plane_eqs[idx].view(1, 4)], dim=0)
                planes_c = torch.cat([planes_c, plane_eqs_c[idx].view(1, 3)], dim=0)
                descs = torch.cat([descs, cur_descs[idx].view(1, self.config.DESC_LEN)], dim=0)

                next_id += 1
            segmentation[segmentation == max_val] = -1

            np.save(planes_filename, planes.cpu().numpy())

            np.save(descs_filename, descs.cpu().numpy())

            # segmentation = (segmentation[:, :, 2] * 256 * 256 +
            #                 segmentation[:, :, 1] * 256 +
            #                 segmentation[:, :, 0]) // 100 - 1
            segmentation_color = torch.zeros((1, 3, target_depth.shape[2], target_depth.shape[3]),
                                             device=dev,
                                             dtype=torch.uint8)
            segmentation_color[:, 0, :, :] = (segmentation[:, 0, :, :] + 1) * 100 % 256
            segmentation_color[:, 1, :, :] = ((segmentation[:, 0, :, :] + 1) * 100 / 256) % 256
            segmentation_color[:, 2, :, :] = ((segmentation[:, 0, :, :] + 1) * 100 / (256 * 256)) % 256
            segmentation_np = utils.to_numpy_image_s(utils.crop_image_zeros_torch(segmentation_color,
                                                                                  batch['left']['image_metas'][b:b+1, :]))

            segmentation_filename_left = os.path.join(segmentation_left_dir,
                                                      batch['frame_num'][b] + '.png')
            # cv2.imwrite(cv2.cvtColor(segmentation_filename_left, cv2.COLOR_BGR2RGB), segmentation_np)
            cv2.imwrite(segmentation_filename_left, segmentation_np)

            segmentation_filename_right = os.path.join(segmentation_right_dir,
                                                       batch['frame_num'][b] + '.png')
            cv2.imwrite(segmentation_filename_right, np.zeros_like(segmentation_np))

            if save_depth:
                depth_numpy = utils.to_numpy_image_s(utils.crop_image_zeros_torch(depth_np,
                                                                                  batch['left']['image_metas'][b:b+1, :]))
                depth_numpy = np.minimum((depth_numpy * 1000.0), 65535).astype(np.uint16)
                depth_filename_left = os.path.join(depth_left_dir,
                                                   batch['frame_num'][b] + '.png')
                # cv2.imwrite(cv2.cvtColor(segmentation_filename_left, cv2.COLOR_BGR2RGB), segmentation_np)
                cv2.imwrite(depth_filename_left, depth_numpy)

                depth_stddev = disp_stddev * depth_np.square() / (fx * baseline)
                depth_stddev_numpy = utils.to_numpy_image_s(utils.crop_image_zeros_torch(depth_stddev,
                                                                                         batch['left']['image_metas'][
                                                                                         b:b + 1, :]))
                depth_stddev_numpy = np.minimum((depth_stddev_numpy * 1000.0), 65535).astype(np.uint16)
                depth_stddev_filename_left = os.path.join(depth_stddev_left_dir,
                                                          batch['frame_num'][b] + '.png')
                # cv2.imwrite(cv2.cvtColor(segmentation_filename_left, cv2.COLOR_BGR2RGB), segmentation_np)
                cv2.imwrite(depth_stddev_filename_left, depth_stddev_numpy)
            if save_det:
                det_image_numpy = utils.to_numpy_image_s(utils.crop_image_zeros_torch(det_images[b:b + 1, :, :, :],
                                                                                      batch['left']['image_metas'][b:b + 1,:]))
                det_image_filename_left = os.path.join(det_left_dir,
                                                       batch['frame_num'][b] + '.jpg')
                # cv2.imwrite(cv2.cvtColor(segmentation_filename_left, cv2.COLOR_BGR2RGB), segmentation_np)
                cv2.imwrite(det_image_filename_left, cv2.cvtColor(det_image_numpy, cv2.COLOR_RGB2BGR))

    def forward(self, molded_images_l, molded_images_r, image_metas, camera, use_refinement=False, gt_depth=None):
        dtype = molded_images_l.dtype
        dev = molded_images_l.device

        fx = camera[:, 0]
        baseline = camera[:, 6]

        if self.training:
            ## Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        if self.detect:
            ## Feature extraction
            [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images_l)
            ## Note that P6 is used in RPN, but not in the classifier heads.

            rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
            mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

            feature_maps = [feature_map for index, feature_map in enumerate(rpn_feature_maps[::-1])]

            ranges = utils.get_ranges_pad_torch_batch(camera)
            ranges_inter = torch.nn.functional.interpolate(ranges, size=(160, 160), mode='bilinear')
            ranges_feat = self.range_conv(ranges_inter * 10)

            ## Loop through pyramid layers
            layer_outputs = []  ## list of lists
            for p in rpn_feature_maps:
                layer_outputs.append(self.rpn(p))

            ## Concatenate layer outputs
            ## Convert from list of lists of level outputs to list of lists
            ## of outputs across levels.
            ## e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
            outputs = list(zip(*layer_outputs))
            outputs = [torch.cat(list(o), dim=1) for o in outputs]
            rpn_class_logits, rpn_class, rpn_bbox = outputs

            ## Generate proposals
            ## Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
            ## and zero padded.
            proposal_count = self.config.POST_NMS_ROIS_TRAINING if self.training and use_refinement == False \
                             else self.config.POST_NMS_ROIS_INFERENCE
            rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                      proposal_count=proposal_count,
                                      nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                      anchors=self.anchors,
                                      config=self.config)
        else:
            mrcnn_feature_maps = torch.tensor([], device=dev, dtype=dtype)
            rpn_class_logits = torch.tensor([], device=dev, dtype=dtype)
            rpn_class = torch.tensor([], device=dev, dtype=dtype)
            rpn_bbox = torch.tensor([], device=dev, dtype=dtype)
            rpn_rois = torch.tensor([], device=dev, dtype=dtype)
            ranges_feat = torch.tensor([], device=dev, dtype=dtype)

        disp1_np, disp2_np, disp3_np, disp3_np_std_dev, normal_feat, normal_np = self.depth(self.config,
                                                                                            camera,
                                                                                            molded_images_l,
                                                                                            molded_images_r)
        depth_np = fx[:, None, None, None] * baseline[:, None, None, None] / torch.clamp(disp3_np, min=1.0e-4)

        return [mrcnn_feature_maps,
                rpn_class_logits,
                rpn_class,
                rpn_bbox,
                rpn_rois,
                ranges_feat,
                normal_feat,
                depth_np,
                disp1_np,
                disp2_np,
                disp3_np,
                disp3_np_std_dev,
                normal_np]

    def comp_disp_loss(self, disp1_np, disp2_np, disp3_np, target_disp, batch_idx):
        disp_mask = target_disp < self.config.MAXDISP

        # disp_np_loss = 0.5 * F.smooth_l1_loss(disp1_np[disp_mask], target_disp[disp_mask], reduction='mean') + \
        #                0.7 * F.smooth_l1_loss(disp2_np[disp_mask], target_disp[disp_mask], reduction='mean') + \
        #                F.smooth_l1_loss(disp3_np[disp_mask], target_disp[disp_mask], reduction='mean')
        disp_np_loss = F.smooth_l1_loss(disp3_np[disp_mask], target_disp[disp_mask], reduction='mean')

        return disp_np_loss

    def comp_normal_loss(self, camera, normal_np, target_normal, target_depth, disp, batch_idx):
        diff_x, diff_y = utils.calc_derivative(target_depth)
        # relative
        diff_x = diff_x / torch.clamp(target_depth, min=1e-3)
        diff_y = diff_y / torch.clamp(target_depth, min=1e-3)

        diff_x = F.max_pool2d(diff_x,
                              (7, 7),
                              stride=1,
                              padding=3)
        diff_y = F.max_pool2d(diff_y,
                              (7, 7),
                              stride=1,
                              padding=3)
        normal_mask = torch.logical_and(diff_x < 0.08, diff_y < 0.08). \
            logical_and((target_normal.norm(dim=1, keepdim=True) - 1.0).abs() < 1.0e-3)

        pred_normal_uvd = normal_np
        pred_normal_xyz = utils.normal_uvd_to_normal_im(camera, self.config, pred_normal_uvd, 320, 320)
        pred_normal_xyz = F.normalize(pred_normal_xyz, dim=1)

        target_normal_uvd = utils.normal_to_normal_uvd_im(camera, self.config, target_normal, 320, 320)
        target_normal_uvd = F.normalize(target_normal_uvd, dim=1)
        # target_normal_xyz_conv = utils.normal_uvd_to_normal_im(camera, self.config, target_normal_uvd, 320, 320)
        # target_normal_xyz_conv = F.normalize(target_normal_xyz_conv, dim=1)
        # target_normal_diff = target_normal - target_normal_xyz_conv

        # angular error for verification
        normal_error_dot = torch.clamp(torch.sum(pred_normal_xyz * target_normal,
                                                 dim=1,
                                                 keepdim=True),
                                       min=-1.0, max=1.0)
        normal_np_ang_error = (torch.acos(normal_error_dot[normal_mask]).abs() * 180.0 / math.pi).mean()

        normal_error = pred_normal_uvd - target_normal_uvd
        if normal_mask.sum() > 0:
            normal_np_loss = F.smooth_l1_loss(pred_normal_uvd[normal_mask.repeat(1, 3, 1, 1)],
                                              target_normal_uvd[normal_mask.repeat(1, 3, 1, 1)])
        else:
            normal_np_ang_error = torch.zeros(1, device=normal_np.device)
            normal_np_loss = torch.zeros(1, device=normal_np.device)

        if self.training:
            prefix = 'train/'
            step = self.global_step
        else:
            prefix = 'val/'
            step = self.global_step + batch_idx

        self.log(prefix + 'normal_np_ang_error', normal_np_ang_error.cpu().detach())

        if batch_idx % self.image_log_rate == 0:
            self.logger.experiment.add_images(prefix + 'normal/normal_mask',
                                              normal_mask.float(),
                                              dataformats='NCHW',
                                              global_step=step)
            self.logger.experiment.add_images(prefix + 'normal/normal_error',
                                              torch.clamp(normal_error.norm(dim=1, keepdim=True), min=0.0, max=1.0),
                                              dataformats='NCHW',
                                              global_step=step)
            self.logger.experiment.add_images(prefix + 'normal/target_normal',
                                              utils.draw_normal_images(target_normal),
                                              dataformats='NCHW',
                                              global_step=step)
            self.logger.experiment.add_images(prefix + 'normal/normal_np',
                                              utils.draw_normal_images(pred_normal_xyz),
                                              dataformats='NCHW',
                                              global_step=step)

        return normal_np_loss

    def log_depth(self, camera, image_metas, image, depth_np, target_disp, target_depth, batch_idx):
        disp_mask = target_disp < self.config.MAXDISP

        if self.training:
            prefix = 'train/'
            step = self.global_step
        else:
            prefix = 'val/'
            step = self.global_step + batch_idx

        if batch_idx % self.image_log_rate == 0:
            self.logger.experiment.add_images(prefix + 'disp/disp_mask',
                                              disp_mask,
                                              dataformats='NCHW',
                                              global_step=step)
            self.logger.experiment.add_images(prefix + 'disp/target_depth',
                                              utils.draw_depth_images(target_depth, max_depth=15),
                                              dataformats='NCHW',
                                              global_step=step)
            self.logger.experiment.add_images(prefix + 'disp/depth_np',
                                              utils.draw_depth_images(depth_np, max_depth=15),
                                              dataformats='NCHW',
                                              global_step=step)

            # bsize = image.shape[0]
            #
            # rays = utils.get_ranges_pad_torch_batch(camera, depth_np.type())
            # XYZ = rays * depth_np
            # XYZ = utils.crop_image_zeros_torch(XYZ, image_metas)
            # XYZ = XYZ.permute(0, 2, 3, 1)
            #
            # image = utils.crop_image_zeros_torch(image, image_metas)
            # pts_col = image.permute(0, 2, 3, 1).type(torch.uint8)
            #
            # for b in range(bsize):
            #     cur_pts = XYZ[b, :, :, :]
            #     cur_pts_col = pts_col[b, :, :, :]
            #     valid_mask = cur_pts.norm(dim=-1) > 0.2
            #
            #     cur_pts = cur_pts[valid_mask, :].unsqueeze(0)
            #     cur_pts_col = cur_pts_col[valid_mask, :].unsqueeze(0)
            #
            #     point_size_config = {
            #         'material': {
            #             'cls': 'PointsMaterial',
            #             'size': 0.05
            #         }
            #     }
            #     self.logger.experiment.add_mesh(prefix + 'disp/point_cloud_' + str(b),
            #                                     vertices=cur_pts,
            #                                     colors=cur_pts_col,
            #                                     config_dict=point_size_config,
            #                                     global_step=step)
            #     # self.logger.experiment.flush()

    def training_step(self, batch, batch_idx):
        # images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks,
        # gt_parameters, gt_depth, extrinsics, gt_plane, gt_segmentation, plane_indices
        camera = batch['camera']

        images_l = batch['left']['image']
        molded_images_l = utils.mold_image_torch(images_l, self.config)
        image_metas = batch['left']['image_metas']
        rpn_match = batch['left']['rpn_match']
        rpn_bbox = batch['left']['rpn_bbox']
        gt_class_ids = batch['left']['gt_class_ids'] % self.config.NUM_CLASSES
        gt_plane_ids = batch['left']['gt_class_ids'] // self.config.NUM_CLASSES
        gt_boxes = batch['left']['gt_boxes']
        gt_masks = batch['left']['gt_masks']
        gt_parameters = batch['left']['gt_parameters']
        gt_depth = batch['left']['depth'][:, 0:1, :, :]
        gt_normal = batch['left']['depth'][:, 1:, :, :]

        images_r = batch['right']['image']
        molded_images_r = utils.mold_image_torch(images_r, self.config)

        bsize = molded_images_l.shape[0]

        dtype = molded_images_l.dtype
        dev = molded_images_l.device

        use_nms = 1
        use_refinement = False

        fx = camera[:, 0]
        baseline = camera[:, 6]

        [mrcnn_feature_maps,
         rpn_class_logits,
         rpn_class,
         rpn_pred_bbox,
         rpn_rois,
         ranges_feat,
         normal_feat,
         depth_np,
         disp1_np,
         disp2_np,
         disp3_np,
         disp3_np_std_dev,
         normal_np] = self.forward(molded_images_l,
                                       molded_images_r,
                                       image_metas,
                                       camera,
                                       use_refinement=use_refinement,
                                       gt_depth=None)

        ## Normalize coordinates
        h, w = self.config.IMAGE_SHAPE[:2]
        scale = torch.tensor([h, w, h, w], device=dev, dtype=dtype)
        gt_boxes_norm = gt_boxes / scale

        if self.detect:
            ## Generate detection targets
            ## Subsamples proposals and generates target outputs for training
            ## Note that proposal class IDs, gt_boxes, and gt_masks are zero
            ## padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask, target_plane_params, target_plane_ids = \
                detection_target_layer(rpn_rois,
                                       gt_class_ids,
                                       gt_plane_ids,
                                       gt_boxes_norm,
                                       gt_masks,
                                       gt_parameters,
                                       self.config,
                                       camera)

            ## Network Heads
            ## Proposal classifier and BBox regressor heads
            # print([maps.shape for maps in mrcnn_feature_maps], target_parameters.shape)
            [mrcnn_class_logits,
             mrcnn_class,
             mrcnn_bbox,
             mrcnn_plane_id_desc,
             mrcnn_plane_id_logits] = self.classifier(mrcnn_feature_maps,
                                                     rois,
                                                     ranges_feat,
                                                     pool_features=False)

            ## Create masks for detections
            mrcnn_mask, _ = self.mask(mrcnn_feature_maps, rois)

            target_mask_pool = torch.nn.functional.interpolate(torch.where(target_mask > 0, 1.0, 0.0),
                                                               size=(self.config.PARAM_POOL_SIZE,
                                                                     self.config.PARAM_POOL_SIZE),
                                                               mode='area')
            mrcnn_plane_params = self.plane_params(normal_feat,
                                                   rois,
                                                   target_mask_pool)

            [rpn_class_loss,
             rpn_bbox_loss,
             mrcnn_class_loss,
             mrcnn_bbox_loss,
             mrcnn_mask_loss,
             mrcnn_plane_params_loss,
             mrcnn_plane_ids_loss] = compute_losses(self.config, camera,
                                                       rpn_match, rpn_bbox,
                                                       rpn_class_logits, rpn_pred_bbox,
                                                       target_class_ids, mrcnn_class_logits,
                                                       target_deltas, mrcnn_bbox,
                                                       target_mask, mrcnn_mask,
                                                       target_plane_params, mrcnn_plane_params,
                                                       target_plane_ids, mrcnn_plane_id_logits)

            maskrcnn_loss = rpn_class_loss + \
                            rpn_bbox_loss + \
                            mrcnn_class_loss + \
                            mrcnn_bbox_loss + \
                            mrcnn_mask_loss + \
                            mrcnn_plane_params_loss + \
                            mrcnn_plane_ids_loss

            self.log('train/rpn_class_loss', rpn_class_loss.cpu().detach())
            self.log('train/rpn_bbox_loss', rpn_bbox_loss.cpu().detach())
            self.log('train/mrcnn_class_loss', mrcnn_class_loss.cpu().detach())
            self.log('train/mrcnn_bbox_loss', mrcnn_bbox_loss.cpu().detach())
            self.log('train/mrcnn_mask_loss', mrcnn_mask_loss.cpu().detach())
            self.log('train/mrcnn_plane_params_loss', mrcnn_plane_params_loss.cpu().detach())
            self.log('train/mrcnn_plane_ids_loss', mrcnn_plane_ids_loss.cpu().detach())
        else:
            maskrcnn_loss = torch.zeros(1, device=dev, dtype=dtype)
        # losses += [rpn_class_loss + rpn_bbox_loss + \
        #            mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_support_loss + mrcnn_support_class_loss]
        # losses += [rpn_class_loss + rpn_bbox_loss +
        #            mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + mrcnn_parameter_loss]
        # losses += [rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss]

        # computing disparity loss
        gt_disp = fx[:, None, None, None] * baseline[:, None, None, None] / torch.clamp(gt_depth, min=1.0e-4)
        disp_np_loss = self.comp_disp_loss(disp1_np,
                                           disp2_np,
                                           disp3_np,
                                           gt_disp,
                                           batch_idx)

        # computing normal loss
        normal_np_loss = self.comp_normal_loss(camera, normal_np, gt_normal, gt_depth, disp3_np, batch_idx)

        self.log_depth(camera, image_metas, images_l, depth_np, gt_disp, gt_depth, batch_idx)

        self.log('train/maskrcnn_loss', maskrcnn_loss.cpu().detach())
        self.log('train/disp_np_loss', (self.options.dispWeight * disp_np_loss).cpu().detach())
        self.log('train/normal_np_loss', (self.options.normWeight * normal_np_loss).cpu().detach())

        if batch_idx % self.image_log_rate == 0:
            self.logger.experiment.add_images('train/image_l',
                                              images_l.type(torch.uint8),
                                              dataformats='NCHW',
                                              global_step=self.global_step)
            self.logger.experiment.add_images('train/image_r',
                                              images_r.type(torch.uint8),
                                              dataformats='NCHW',
                                              global_step=self.global_step)
            self.logger.experiment.add_images('train/target_instances',
                                              utils.draw_instance_images(self.config,
                                                                         images_l,
                                                                         gt_boxes,
                                                                         gt_masks,
                                                                         gt_class_ids), dataformats='NCHW',
                                              global_step=self.global_step)

        loss = maskrcnn_loss + self.options.dispWeight * disp_np_loss + self.options.normWeight * normal_np_loss
        # loss = disp_np_loss + normal_np_loss

        return loss

    def unmold_detections(self, camera, detections, detection_masks, detection_plane_params, depth_np, normals_np,
                          images=None):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        tensor_type = detections.type()
        dev = detections.device

        if self.config.GLOBAL_MASK:
            masks = detection_masks[torch.arange(len(detection_masks), device=dev, dtype=torch.long), 0, :, :]
        else:
            masks = detection_masks[torch.arange(len(detection_masks), device=dev, dtype=torch.long), detections[:, 4].long(), :, :]
            pass

        final_masks = []
        for detectionIndex in range(len(detections)):
            box = detections[detectionIndex][:4].long()
            if (box[2] - box[0]) * (box[3] - box[1]) <= 0:
                continue

            mask = masks[detectionIndex]
            final_mask = utils.resize_mask_full(self.config, box, mask).unsqueeze(0)
            final_masks.append(final_mask)
            continue

        if len(final_masks) == 0:
            return torch.zeros_like(detections)

        final_masks = torch.stack(final_masks, dim=0)
        num_detections = final_masks.shape[0]

        masks = final_masks

        # # TODO Just for testing HSV descriptors
        # if images is not None:
        #     images_hsv = utils.rgb2hsv(images.unsqueeze(0))
        #     for n in range(num_detections):
        #         cur_mask_det = masks[n, :, :, :] > 0.5
        #         cur_mask_valid = depth_np > 0.2
        #         cur_mask = torch.logical_and(cur_mask_det, cur_mask_valid)
        #         cur_area = cur_mask.sum()
        #
        #         cur_colors = images_hsv[0, :, cur_mask.squeeze(0)]
        #         hist_h = torch.histc(cur_colors[0, :], min=0, max=360, bins=32) / cur_area
        #         hist_s = torch.histc(cur_colors[0, :], min=0, max=256, bins=32) / cur_area
        #
        #         detections[n, 9:9 + 64] = torch.cat([hist_h, hist_s])

        if self.config.ANCHOR_TYPE == 'none_exp_plane_params':
            ranges = utils.get_ranges_pad_torch(camera)
            XYZ = ranges * depth_np

            # add and remove batch dimension
            normals_xyz = F.normalize(utils.normal_uvd_to_normal(camera.view(1, -1),
                                                                 self.config,
                                                                 detection_plane_params.unsqueeze(0),
                                                                 320, 320).squeeze(0), dim=1)

            for n in range(num_detections):
                cur_mask_det = masks[n, :, :, :] > 0.5
                cur_mask_valid = depth_np > 0.2
                cur_mask = torch.logical_and(cur_mask_det, cur_mask_valid)

                if cur_mask.sum() > 500:
                    cur_pts = XYZ[0, :, cur_mask.squeeze(0)]
                    # cur_normal = F.normalize(detection_plane_params[n, :], dim=0)
                    cur_normal = normals_xyz[n, :]

                    # inliers_mask_np, cur_plane_np = utils_cpp_py.ransac_dist(cur_pts.cpu().numpy(),
                    #                                                          cur_normal.cpu().numpy(),
                    #                                                          50,
                    #                                                          0.05,
                    #                                                          False)
                    # inliers_mask = torch.from_numpy(inliers_mask_np, ).type_as(cur_normal)
                    # cur_plane = torch.from_numpy(cur_plane_np).type_as(cur_normal)
                    # cur_plane = cur_plane / cur_plane.norm().clip(min=1.0e-5).square()

                    inliers_mask_np, cur_plane_np = utils_cpp_py.ransac_plane(cur_pts.cpu().numpy(),
                                                                             50,
                                                                             0.05,
                                                                             False)
                    inliers_mask = torch.from_numpy(inliers_mask_np, ).type_as(cur_normal)

                    inliers_points = cur_pts[:, inliers_mask > 0.5]
                    cur_d = (inliers_points * cur_normal.view(3, 1)).sum(dim=0).mean()
                    cur_plane = cur_normal * cur_d

                    # if cur_plane.norm() < 0.2:
                    #     detections[n, 6:9] = 0.0
                    #     continue
                    # inliers_ratio = inliers_mask.sum().float() / cur_pts.shape[1]
                    # print(inliers_ratio)
                    # plane_offset = 1.0 / torch.clamp(cur_plane.norm(), min=1e-4)
                    # plane_normal = cur_plane * plane_offset
                    # plane_eq = torch.cat([plane_normal, -plane_offset[None]])
                    detections[n, 6:9] = cur_plane
                else:
                    detections[n, 6:9] = 0.0
        elif self.config.ANCHOR_TYPE == 'none_exp_mean':
            fx = camera[0]
            fy = camera[1]
            cx = camera[2]
            cy = camera[3]
            w = int(camera[4])
            h = int(camera[5])
            dcy = (w - h) / 2
            a = self.config.UVD_CONST
            baseline = camera[6]

            normals_xyz = utils.normal_uvd_to_normal_im(camera.view(1, -1),
                                                        self.config,
                                                        normals_np.unsqueeze(0),
                                                        320, 320)

            ranges = utils.get_ranges_pad_torch(camera)
            XYZ = ranges * depth_np

            for n in range(num_detections):
                cur_mask_det = masks[n, :, :, :] > 0.5
                cur_mask_valid = depth_np > 0.2
                cur_mask = torch.logical_and(cur_mask_det, cur_mask_valid)

                # if cur_mask.sum() > 500 and float(cur_mask.sum()) / cur_mask_det.sum() > 0.6:
                if cur_mask.sum() > 500:
                    cur_pts = XYZ[0, :, cur_mask.squeeze(0)]
                    cur_normals = normals_xyz[0, :, cur_mask.squeeze(0)]

                    cur_normal = F.normalize(cur_normals.mean(dim=1), dim=0)

                    inliers_mask_np, cur_plane_np = utils_cpp_py.ransac_dist(cur_pts.cpu().numpy(),
                                                                             cur_normal.cpu().numpy(),
                                                                             50,
                                                                             0.05,
                                                                             False)
                    inliers_mask = torch.from_numpy(inliers_mask_np).type_as(cur_normal)
                    cur_plane = torch.from_numpy(cur_plane_np).type_as(cur_normal)

                    inliers_ratio = inliers_mask.sum().float() / cur_pts.shape[1]
                    # print(inliers_ratio)
                    plane_offset = 1.0 / torch.clamp(cur_plane.norm(), min=1e-4)
                    plane_normal = cur_plane * plane_offset
                    # plane_eq = torch.cat([plane_normal, -plane_offset[None]])
                    detections[n, 6:9] = cur_plane * plane_offset.square()

                else:
                    detections[n, 6:9] = 0.0

        elif self.config.ANCHOR_TYPE == 'none_exp_ransac':

            ranges = utils.get_ranges_pad_torch(camera)
            XYZ = ranges * depth_np

            for n in range(num_detections):
                cur_mask_det = masks[n, :, :, :] > 0.5
                cur_mask_valid = depth_np > 0.2
                cur_mask = torch.logical_and(cur_mask_det, cur_mask_valid)

                # if cur_mask.sum() > 500 and float(cur_mask.sum()) / cur_mask_det.sum() > 0.6:
                if cur_mask.sum() > 500:
                    cur_pts = XYZ[0, :, cur_mask.squeeze(0)]

                    inlier_mask_ransac_np, plane_ransac_np = utils_cpp_py.ransac_plane(cur_pts.cpu().numpy(),
                                                                                      200,
                                                                                      0.05,
                                                                                      False)
                    inlier_mask_ransac = torch.from_numpy(inlier_mask_ransac_np).type_as(detections)
                    plane_ransac = torch.from_numpy(plane_ransac_np).type_as(detections)
                    plane_ransac = plane_ransac / plane_ransac.norm().clip(min=1.0e-5).square()

                    detections[n, 6:9] = plane_ransac

                else:
                    detections[n, 6:9] = 0.0

        return detections

    def detection_step(self, batch, batch_idx):
        # images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks,
        # gt_parameters, gt_depth, extrinsics, gt_plane, gt_segmentation, plane_indices
        camera = batch['camera']

        images_l = batch['left']['image']
        molded_images_l = utils.mold_image_torch(images_l, self.config)
        image_metas = batch['left']['image_metas']
        rpn_match = batch['left']['rpn_match']
        rpn_bbox = batch['left']['rpn_bbox']
        gt_class_ids = batch['left']['gt_class_ids'] % self.config.NUM_CLASSES
        gt_plane_ids = batch['left']['gt_class_ids'] // self.config.NUM_CLASSES
        gt_boxes = batch['left']['gt_boxes']
        gt_masks = batch['left']['gt_masks']
        gt_parameters = batch['left']['gt_parameters']
        gt_depth = batch['left']['depth'][:, 0:1, :, :]
        gt_normal = batch['left']['depth'][:, 1:, :, :]

        images_r = batch['right']['image']
        molded_images_r = utils.mold_image_torch(images_r, self.config)

        bsize = molded_images_l.shape[0]

        use_nms = 1
        use_refinement = False

        # tensor_type = molded_images_l.type()
        dtype = molded_images_l.dtype
        dev = molded_images_l.device

        fx = camera[:, 0]
        baseline = camera[:, 6]

        [mrcnn_feature_maps,
         rpn_class_logits,
         rpn_class,
         rpn_pred_bbox,
         rpn_rois,
         ranges_feat,
         normal_feat,
         depth_np,
         disp1_np,
         disp2_np,
         disp3_np,
         disp3_np_std_dev,
         normal_np] = self.forward(molded_images_l,
                                       molded_images_r,
                                       image_metas,
                                       camera,
                                       use_refinement=use_refinement,
                                       gt_depth=None)

        # poor_disp_mask = disp3_np_std_dev > 5.0
        # depth_np_masked = depth_np.clone()
        # depth_np_masked[poor_disp_mask] = 0

        ## Normalize coordinates
        h, w = self.config.IMAGE_SHAPE[:2]
        scale = torch.tensor([h, w, h, w], device=dev, dtype=dtype)
        gt_boxes_norm = gt_boxes / scale

        if self.detect:
            if self.annotations_as_detections:
                ## Network Heads
                ## Proposal classifier and BBox regressor heads
                [mrcnn_class_logits,
                 mrcnn_class,
                 mrcnn_bbox,
                 mrcnn_plane_id_desc,
                 mrcnn_plane_id_logits] = self.classifier(mrcnn_feature_maps,
                                                          gt_boxes_norm,
                                                          ranges_feat,
                                                          pool_features=False)

                nsize = mrcnn_class.shape[1]
                csize = mrcnn_class.shape[2]

                # gt_class = torch.zeros_like(mrcnn_class)
                # idxs = torch.arange(bsize * nsize, dtype=torch.long, device=dev)
                # idxs_class = gt_class_ids.view(bsize * nsize).long()

                # gt_class = gt_class.view(bsize * nsize, csize)
                # gt_class[idxs, idxs_class] = 1.0
                # gt_class = gt_class.view(bsize, nsize, csize)

                # class_parameters = torch.zeros(bsize, nsize, self.config.NUM_PARAMETERS, device=dev)
                # class_parameters = class_parameters.view(bsize * nsize, self.config.NUM_PARAMETERS)
                # class_parameters = mrcnn_parameters.view(bsize * nsize, csize, self.config.NUM_PARAMETERS)[idxs, idxs_class, :]
                # class_parameters = self.config.applyAnchorsTensor(idxs_class, class_parameters)
                # class_parameters = class_parameters.view(bsize, nsize, self.config.NUM_PARAMETERS)

                ## Detections
                ## output is [batch, num_detections, (y1, x1, y2, x2, class_id, score, plane params, desc)] in image coordinates
                molded_detections = torch.cat([gt_boxes,
                                               gt_class_ids.float().unsqueeze(-1),
                                               torch.ones_like(gt_class_ids.float().unsqueeze(-1)),
                                               torch.zeros(bsize, nsize, self.config.NUM_PARAMETERS, device=dev),
                                               mrcnn_plane_id_desc],
                                              dim=-1)
            else:
                ## Network Heads
                ## Proposal classifier and BBox regressor heads
                [mrcnn_class_logits,
                 mrcnn_class,
                 mrcnn_bbox,
                 mrcnn_plane_id_desc,
                 mrcnn_plane_id_logits] = self.classifier(mrcnn_feature_maps,
                                                          rpn_rois,
                                                          ranges_feat,
                                                          pool_features=False)

                ## Detections
                ## output is [batch, num_detections, (y1, x1, y2, x2, class_id, score, plane params, desc)] in image coordinates
                molded_detections = detection_layer(self.config,
                                                    rpn_rois,
                                                    mrcnn_class,
                                                    mrcnn_bbox,
                                                    mrcnn_plane_id_desc,
                                                    image_metas,
                                                    use_nms=use_nms)

                nsize = molded_detections.shape[1]

            ## Convert boxes to normalized coordinates
            ## TODO: let DetectionLayer return normalized coordinates to avoid
            ##       unnecessary conversions
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = torch.tensor([h, w, h, w], device=dev)
            detection_boxes = molded_detections[:, :, :4] / scale

            ## Create masks for detections
            mrcnn_mask, roi_features = self.mask(mrcnn_feature_maps, detection_boxes)

            mrcnn_mask_pool = torch.zeros((mrcnn_mask.shape[0],
                                           mrcnn_mask.shape[1],
                                           self.config.PARAM_POOL_SIZE,
                                           self.config.PARAM_POOL_SIZE),
                                           device=dev, dtype=dtype)
            for b in range(bsize):
                idxs = torch.arange(nsize, dtype=torch.long, device=dev)
                idxs_class = molded_detections[b, :, 4].long()
                mrcnn_mask_pool[b, :, :, :] = torch.nn.functional.interpolate(mrcnn_mask[b, idxs, idxs_class, :, :].unsqueeze(1),
                                                                                 size=(self.config.PARAM_POOL_SIZE,
                                                                                       self.config.PARAM_POOL_SIZE),
                                                                                 mode='area').squeeze(1)
            detection_plane_params = self.plane_params(normal_feat,
                                                       detection_boxes,
                                                       mrcnn_mask_pool)

            if self.annotations_as_detections:
                nsize = mrcnn_class.shape[1]
                csize = mrcnn_class.shape[2]
                msize = gt_masks.shape[2]
                gt_masks_class = torch.zeros((bsize,
                                              nsize,
                                              csize,
                                              msize,
                                              msize),
                                             device=dev)
                idxs = torch.arange(bsize * nsize, dtype=torch.long, device=dev)
                idxs_class = gt_class_ids.view(bsize * nsize).long()

                gt_masks_class = gt_masks_class.view(bsize * nsize, csize, msize, msize)
                gt_masks_class[idxs, idxs_class, :, :] = gt_masks.view(bsize * nsize, msize, msize)
                gt_masks_class = gt_masks_class.view(bsize, nsize, csize, msize, msize)
                mrcnn_mask = gt_masks_class
            else:
                ## Create masks for detections
                mrcnn_mask, roi_features = self.mask(mrcnn_feature_maps, detection_boxes)

            detections = torch.zeros(molded_detections.shape, device=dev, dtype=dtype)

            for b in range(bsize):
                cur_detections = self.unmold_detections(camera[b, :],
                                                        molded_detections[b, :, :],
                                                        mrcnn_mask[b, :, :, :, :],
                                                        detection_plane_params[b, :, :],
                                                        depth_np[b, :, :, :],
                                                        normal_np[b, :, :, :],
                                                        images=images_l[b, :, :, :])
                detections[b, :, :] = cur_detections
                # detection_masks[b, :, :, :] = cur_detection_masks
        else:
            detections = torch.zeros((bsize,
                                      self.config.DETECTION_MAX_INSTANCES,
                                      4 + 2 + self.config.NUM_PARAMETERS + self.config.DESC_LEN),
                                     device=dev, dtype=dtype)
            mrcnn_mask = torch.zeros((bsize,
                                      self.config.DETECTION_MAX_INSTANCES,
                                      self.config.NUM_CLASSES,
                                      self.config.MASK_SHAPE[0],
                                      self.config.MASK_SHAPE[0]), device=dev, dtype=dtype)

        # computing disparity loss
        gt_disp = fx[:, None, None, None] * baseline[:, None, None, None] / torch.clamp(gt_depth, min=1.0e-4)
        disp_np_loss = self.comp_disp_loss(disp1_np,
                                           disp2_np,
                                           disp3_np,
                                           gt_disp,
                                           batch_idx)

        # computing normal loss
        normal_np_loss = self.comp_normal_loss(camera, normal_np, gt_normal, gt_depth, disp3_np, batch_idx)

        self.log_depth(camera, image_metas, images_l, depth_np, gt_disp, gt_depth, batch_idx)

        self.log('val/disp_np_loss', (self.options.dispWeight * disp_np_loss).cpu().detach())
        self.log('val/normal_np_loss', (self.options.normWeight * normal_np_loss).cpu().detach())

        if batch_idx % self.image_log_rate == 0:
            self.logger.experiment.add_images('val/image_l',
                                              images_l.type(torch.uint8),
                                              dataformats='NCHW',
                                              global_step=self.global_step + batch_idx)
            self.logger.experiment.add_images('val/image_r',
                                              images_r.type(torch.uint8),
                                              dataformats='NCHW',
                                              global_step=self.global_step + batch_idx)
            self.logger.experiment.add_images('val/det_instances',
                                              utils.draw_instance_images(self.config,
                                                                         images_l,
                                                                         detections[:, :, 0:4],
                                                                         mrcnn_mask,
                                                                         detections[:, :, 4]), dataformats='NCHW',
                                              global_step=self.global_step + batch_idx)
            self.logger.experiment.add_images('val/target_instances',
                                              utils.draw_instance_images(self.config,
                                                                         images_l,
                                                                         gt_boxes,
                                                                         gt_masks,
                                                                         gt_class_ids), dataformats='NCHW',
                                              global_step=self.global_step + batch_idx)

            normals_pl = utils.detections_to_normal(self.config,
                                                    images_l,
                                                    detections[:, :, 0:4],
                                                    mrcnn_mask,
                                                    detections[:, :, 4],
                                                    detections[:, :, 6:9])
            self.logger.experiment.add_images('val/normal/normal_pl',
                                              utils.draw_normal_images(-normals_pl),
                                              dataformats='NCHW',
                                              global_step=self.global_step + batch_idx)
            self.logger.experiment.add_images('val/normal/target_normal',
                                              utils.draw_normal_images(gt_normal),
                                              dataformats='NCHW',
                                              global_step=self.global_step + batch_idx)

        return detections, mrcnn_mask, disp3_np, disp3_np_std_dev, depth_np, normal_np

    def validation_step_common(self, batch, batch_idx):
        camera = batch['camera']

        images_l = batch['left']['image']
        molded_images_l = utils.mold_image_torch(images_l, self.config)
        image_metas = batch['left']['image_metas']
        rpn_match = batch['left']['rpn_match']
        rpn_bbox = batch['left']['rpn_bbox']
        gt_class_ids = batch['left']['gt_class_ids'] % self.config.NUM_CLASSES
        gt_plane_ids = batch['left']['gt_class_ids'] // self.config.NUM_CLASSES
        gt_boxes = batch['left']['gt_boxes']
        gt_masks = batch['left']['gt_masks']
        gt_parameters = batch['left']['gt_parameters']
        gt_depth = batch['left']['depth'][:, 0:1, :, :]
        gt_normal = batch['left']['depth'][:, 1:, :, :]

        images_r = batch['right']['image']
        molded_images_r = utils.mold_image_torch(images_r, self.config)

        bsize = molded_images_l.shape[0]

        tensor_type = molded_images_l.type()

        with self.profiler.profile('detection'):
            detections, masks, disp_np, disp_stddev, depth_np, normal_np = self.detection_step(batch, batch_idx)

        [error_dist_hist,
         error_norm_hist,
         error_area_hist,
         error_cnt_hist,
         target_dist_hist] = utils.evaluate_plane_dist_norm(self.config,
                                                            camera,
                                                            detections,
                                                            masks,
                                                            gt_depth,
                                                            max_bins=self.config.EVALUATION_BINS,
                                                            depth_pred=depth_np,
                                                            batch=batch,
                                                            profiler=self.profiler)

        if self.export_detections:
            self.save_as_annotation(batch,
                                    batch_idx,
                                    camera,
                                    detections,
                                    masks,
                                    gt_depth,
                                    depth_np,
                                    disp_stddev,
                                    from_target=False,
                                    save_depth=True,
                                    save_det=True,
                                    output_dir='annotation_plane_params_det')

        # if batch_idx % self.image_log_rate == 0:
        #     dist_error = (error_dist_hist.sum() / error_area_hist.sum().clamp(min=1.0)).sqrt()
        #     norm_error = (error_norm_hist.sum() / error_area_hist.sum().clamp(min=1.0)).sqrt()
        #     with open('error.log', 'a') as ef:
        #         ef.write('%6d %.3f %.3f %s %s\n' % (batch_idx,
        #                                             norm_error,
        #                                             dist_error,
        #                                             batch['scene_id'],
        #                                             batch['frame_num']))

        if self.evaluate_descriptors:
            self.dist_hist(error_dist_hist, error_area_hist, error_cnt_hist)
            self.norm_hist(error_norm_hist, error_area_hist, error_cnt_hist)
            self.target_dist_hist(target_dist_hist, error_area_hist, error_cnt_hist)

            det_plane_ids, det_mask_areas = utils.find_plane_ids(self.config,
                                                                 detections[:, :, 0:4],
                                                                 masks,
                                                                 detections[:, :, 4],
                                                                 gt_boxes,
                                                                 gt_masks,
                                                                 gt_class_ids,
                                                                 gt_plane_ids)

            valid_idxs = det_plane_ids >= 0

            frame_nums = torch.from_numpy(np.asarray(batch['frame_num']).astype(np.float32)).type(tensor_type)
            frame_nums = frame_nums.view(bsize, 1).expand(-1, det_plane_ids.shape[1])
            # timestamps
            tss = frame_nums / self.fps
            cat_idxs = torch.minimum((det_mask_areas.sqrt() / 50).long(),
                                     torch.tensor(5, dtype=torch.long, device=det_mask_areas.device))

            scene_ids = batch['scene_id']
            scene_ids_valid = []
            for b in range(bsize):
                n_valid_batch = valid_idxs[b, :].sum()
                scene_ids_valid.extend([scene_ids[b]] * int(n_valid_batch))

            self.desc_ranks(det_plane_ids[valid_idxs],
                            detections[valid_idxs, :][:, 9:],
                            tss[valid_idxs],
                            cat_idxs[valid_idxs],
                            scene_ids_valid)

    def validation_step(self, batch, batch_idx):
        self.validation_step_common(batch, batch_idx)

    def on_validation_epoch_end(self):
        [error_dist_hist_val,
         error_dist_hist_conf_int,
         _,
         _,
         error_dist_val] = self.dist_hist.compute()
        [error_norm_hist_val,
         error_norm_hist_conf_int,
         error_hist_area,
         error_hist_cnt,
         error_norm_val] = self.norm_hist.compute()
        [target_dist_hist_val,
         _,
         _,
         target_dist_val] = self.target_dist_hist.compute()

        rank_q0, rank_q1, rank_q2, rank_q3, rank_q4, rank_mean = self.desc_ranks.compute()

        self.dist_hist.reset()
        self.norm_hist.reset()
        self.target_dist_hist.reset()
        self.desc_ranks.reset()

        self.log('val/mean_error_dist', error_dist_val.cpu().detach())
        self.log('val/mean_error_norm', error_norm_val.cpu().detach())
        self.log('val/mean_target_dist', target_dist_val.cpu().detach())

        for b in range(error_dist_hist_val.shape[0]):
            self.log('val/hist_error_dist %02d' % b, error_dist_hist_val[b].cpu().detach())
        for b in range(error_norm_hist_val.shape[0]):
            self.log('val/hist_error_norm %02d' % b, error_norm_hist_val[b].cpu().detach())
        for b in range(error_hist_cnt.shape[0]):
            self.log('val/error cnt %02d' % b, error_hist_cnt[b].cpu().detach())
        for b in range(error_hist_area.shape[0]):
            self.log('val/error area %02d' % b, error_hist_area[b].cpu().detach())
        for b in range(target_dist_hist_val.shape[0]):
            self.log('val/hist_target_dist %02d' % b, target_dist_hist_val[b].cpu().detach())

        for b in range(rank_q0.shape[0]):
            self.log('val/rank_q0 %02d' % b, rank_q0[b].cpu().detach())
            self.log('val/rank_q1 %02d' % b, rank_q1[b].cpu().detach())
            self.log('val/rank_q2 %02d' % b, rank_q2[b].cpu().detach())
            self.log('val/rank_q3 %02d' % b, rank_q3[b].cpu().detach())
            self.log('val/rank_q4 %02d' % b, rank_q4[b].cpu().detach())
            self.log('val/rank_mean %02d' % b, rank_mean[b].cpu().detach())

    def test_step(self, batch, batch_idx):
        self.validation_step_common(batch, batch_idx)

    def on_test_epoch_end(self):
        [error_dist_hist_val,
         error_dist_hist_conf_int,
         _,
         _,
         error_dist_val] = self.dist_hist.compute()
        [error_norm_hist_val,
         error_norm_hist_conf_int,
         error_hist_area,
         error_hist_cnt,
         error_norm_val] = self.norm_hist.compute()
        [target_dist_hist_val,
         _,
         _,
         target_dist_val] = self.target_dist_hist.compute()

        rank_q0, rank_q1, rank_q2, rank_q3, rank_q4, rank_mean = self.desc_ranks.compute()

        self.dist_hist.reset()
        self.norm_hist.reset()
        self.target_dist_hist.reset()
        self.desc_ranks.reset()

        self.log('val/mean_error_dist', error_dist_val.cpu().detach())
        self.log('val/mean_error_norm', error_norm_val.cpu().detach())
        self.log('val/mean_target_dist', target_dist_val.cpu().detach())

        for b in range(error_dist_hist_val.shape[0]):
            self.log('val/hist_error_dist %02d' % b, error_dist_hist_val[b].cpu().detach())
        for b in range(error_norm_hist_val.shape[0]):
            self.log('val/hist_error_norm %02d' % b, error_norm_hist_val[b].cpu().detach())
        for b in range(error_hist_cnt.shape[0]):
            self.log('val/error cnt %02d' % b, error_hist_cnt[b].cpu().detach())
        for b in range(error_hist_area.shape[0]):
            self.log('val/error area %02d' % b, error_hist_area[b].cpu().detach())
        for b in range(target_dist_hist_val.shape[0]):
            self.log('val/hist_target_dist %02d' % b, target_dist_hist_val[b].cpu().detach())

        for b in range(rank_q0.shape[0]):
            self.log('val/rank_q0 %02d' % b, rank_q0[b].cpu().detach())
            self.log('val/rank_q1 %02d' % b, rank_q1[b].cpu().detach())
            self.log('val/rank_q2 %02d' % b, rank_q2[b].cpu().detach())
            self.log('val/rank_q3 %02d' % b, rank_q3[b].cpu().detach())
            self.log('val/rank_q4 %02d' % b, rank_q4[b].cpu().detach())
            self.log('val/rank_mean %02d' % b, rank_mean[b].cpu().detach())

        dist_hist_str = ''
        dist_hist_min_str = ''
        dist_hist_max_str = ''
        for b in range(error_dist_hist_val.shape[0]):
            dist_hist_str += '%.3f ' % float(error_dist_hist_val[b])
            dist_hist_min_str += '%.3f ' % float(error_dist_hist_conf_int[b, 0])
            dist_hist_max_str += '%.3f ' % float(error_dist_hist_conf_int[b, 1])

        norm_hist_str = ''
        norm_hist_min_str = ''
        norm_hist_max_str = ''
        for b in range(error_norm_hist_val.shape[0]):
            norm_hist_str += '%.3f ' % float(error_norm_hist_val[b])
            norm_hist_min_str += '%.3f ' % float(error_norm_hist_conf_int[b, 0])
            norm_hist_max_str += '%.3f ' % float(error_norm_hist_conf_int[b, 1])

        target_dist_hist_str = ''
        for b in range(target_dist_hist_val.shape[0]):
            target_dist_hist_str += '%.3f ' % float(target_dist_hist_val[b])

        cnt_str = ''
        for b in range(error_hist_cnt.shape[0]):
            cnt_str += str(int(error_hist_cnt[b])) + ' '
        area_str = ''
        for b in range(error_hist_area.shape[0]):
            area_str += str(int(error_hist_area[b])) + ' '

        rank_mean_str = ''
        for b in range(rank_mean.shape[0]):
            rank_mean_str += '%.3f' % rank_mean[b] + ' '
        rank_q0_str = ''
        for b in range(rank_q0.shape[0]):
            rank_q0_str += str(int(rank_q0[b])) + ' '
        rank_q1_str = ''
        for b in range(rank_q1.shape[0]):
            rank_q1_str += str(int(rank_q1[b])) + ' '
        rank_q2_str = ''
        for b in range(rank_q2.shape[0]):
            rank_q2_str += str(int(rank_q2[b])) + ' '
        rank_q3_str = ''
        for b in range(rank_q3.shape[0]):
            rank_q3_str += str(int(rank_q3[b])) + ' '
        rank_q4_str = ''
        for b in range(rank_q4.shape[0]):
            rank_q4_str += str(int(rank_q4[b])) + ' '

        print('%.3f ' % float(error_dist_val))
        print(dist_hist_str)
        print(dist_hist_min_str)
        print(dist_hist_max_str)
        print('%.3f ' % float(error_norm_val))
        print(norm_hist_str)
        print(norm_hist_min_str)
        print(norm_hist_max_str)
        print(cnt_str)
        print(area_str)
        print('%.3f ' % float(target_dist_val))
        print(target_dist_hist_str)
        print(rank_mean_str)
        print(rank_q0_str)
        print(rank_q1_str)
        print(rank_q2_str)
        print(rank_q3_str)
        print(rank_q4_str)
