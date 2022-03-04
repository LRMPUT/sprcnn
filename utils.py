"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import os
import time
import math
import random
import numpy as np
import cv2

import utils

cv2.setNumThreads(0)
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning.profiler import PassThroughProfiler

import itertools
from skimage.measure import find_contours

import utils_cpp_py


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        ## Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            ## x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            ## No mask for this instance. Might happen due to
            ## resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    ## Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    ## Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    ## Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    ## Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dy, dx, dh, dw], dim=1)
    return result


############################################################
#  Dataset
############################################################
def resize_image(image, min_dim=None, max_dim=None, padding=False, interp='bilinear'):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    ## Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    ## Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    ## Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    ## Resize image and mask
    if scale != 1:
        image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
    ## Need padding?
    if padding:
        ## Get new height and width
        h, w = image.shape[:2]
        top_pad = (min_dim - h) // 2
        bottom_pad = min_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        m = cv2.resize(m.astype(np.uint8) * 255, mini_shape)
        mini_mask[:, :, i] = np.where(m >= 128, 1, 0)
    return mini_mask


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    ## Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    ## Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    ## Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    ## Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    ## Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    ## Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    ## Anchors
    ## [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    anchors = np.concatenate(anchors, axis=0)
    return anchors


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


## Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta_torch(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[0]
    image_shape = meta[1:4]
    window = meta[4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[8:]
    return image_id, image_shape, window, active_class_ids


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return (images.astype(np.float32)/config.SCALE_PIXEL - config.MEAN_PIXEL) / config.STD_PIXEL


def mold_image_torch(images, config):
    """Takes a image normalized with mold() and returns the original."""
    # return torch.clamp((normalized_images * config.STD_PIXEL_TENSOR[None, :, None, None] + \
    #          config.MEAN_PIXEL_TENSOR[None, :, None, None]) * config.SCALE_PIXEL, min=0.0, max=255.0)
    return (images / config.SCALE_PIXEL - config.MEAN_PIXEL_TENSOR[None, :, None, None]) / \
            config.STD_PIXEL_TENSOR[None, :, None, None]


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return ((normalized_images * config.STD_PIXEL + config.MEAN_PIXEL) * config.SCALE_PIXEL).astype(np.uint8)


def unmold_image_torch(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    # return torch.clamp((normalized_images * config.STD_PIXEL_TENSOR[None, :, None, None] + \
    #          config.MEAN_PIXEL_TENSOR[None, :, None, None]) * config.SCALE_PIXEL, min=0.0, max=255.0)
    return (normalized_images * config.STD_PIXEL_TENSOR[None, :, None, None] + \
             config.MEAN_PIXEL_TENSOR[None, :, None, None]) * config.SCALE_PIXEL


def pad_image(image, meta, config, val=0):
    im_h = image.shape[0]
    im_w = image.shape[1]
    im_ch = image.shape[2]

    image_id, image_shape, window, active_class_ids = parse_image_meta(meta)

    # all windows have to be the same
    image_pad = val * np.ones((config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, im_ch), dtype=image.dtype)
    image_pad[window[0]: window[2], window[1]: window[3], :] = image

    return image_pad


def pad_image_torch(image, meta, config, val=0):
    im_b = image.shape[0]
    im_ch = image.shape[1]
    im_h = image.shape[2]
    im_w = image.shape[3]

    image_id, image_shape, window, active_class_ids = parse_image_meta_torch(meta)

    # all windows have to be the same
    image_pad = val * torch.ones((im_b, im_ch, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM), device=image.device, dtype=image.dtype)
    image_pad[:, :,  window[0, 0]: window[0, 2], window[0, 1]: window[0, 3]] = image

    return image_pad


def crop_image_zeros(image, meta):
    image_id, image_shape, window, active_class_ids = parse_image_meta(meta)
    # all windows have to be the same
    return image[window[0]: window[2], window[1]: window[3], :]


def crop_image_zeros_torch(image, meta):
    image_id, image_shape, window, active_class_ids = parse_image_meta_torch(meta)
    # all windows have to be the same
    return image[:, :, window[0, 0]: window[0, 2], window[0, 1]: window[0, 3]]


def pad_zeros(data, N, dim=0):
    zeros_shape = list(data.shape)
    zeros_shape[dim] = N - zeros_shape[dim]

    return np.concatenate([data, np.zeros(zeros_shape, dtype=data.dtype)], axis=dim)


def pad_zeros_torch(data, N, dim=0):
    zeros_shape = list(data.shape)
    zeros_shape[dim] = N - zeros_shape[dim]

    return torch.cat([data, torch.zeros(zeros_shape, device=data.device, dtype=data.dtype)], dim=dim)


## Visualization
class ColorPalette:
    def __init__(self, numColors):
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [128, 0, 255],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [255, 230, 180],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ], dtype=np.uint8)

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3), dtype=np.uint8)], axis=0)
            pass

        return

    def getColorMap(self, returnTuples=False):
        if returnTuples:
            return [tuple(color) for color in self.colorMap.tolist()]
        else:
            return self.colorMap

    def getColor(self, index):
        if index >= self.colorMap.shape[0]:
            return np.random.randint(255, size = (3), dtype=np.uint8)
        else:
            return self.colorMap[index]
        pass


def writePointCloud(filename, point_cloud):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(point_cloud))
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
end_header
"""
        f.write(header)
        for point in point_cloud:
            for valueIndex, value in enumerate(point):
                if valueIndex < 3:
                    f.write(str(value) + ' ')
                else:
                    f.write(str(int(value)) + ' ')
                    pass
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


## The function to compute plane depths from plane parameters
def calcPlaneDepths(planes, width, height, camera, max_depth=20):
    urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (camera[4] + 1) - camera[2]) / camera[0]
    vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (camera[5] + 1) - camera[3]) / camera[1]
    ranges = np.stack([urange, vrange, np.ones(urange.shape)], axis=-1)

    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())
    # normalXYZ[normalXYZ < 1e-4] = 1e-4
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    return planeDepths


def to_numpy_image(image):
    image_np = image.permute(0, 2, 3, 1).cpu().detach().numpy()

    return image_np


def to_torch_image(image_np, dev):
    image = torch.from_numpy(image_np).to(dev).permute(0, 3, 1, 2)

    return image


def to_torch_image_s(image):
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


def to_numpy_image_s(image):
    return image.squeeze(0).permute(1, 2, 0).cpu().numpy()


def draw_depth_images(depth, max_depth=10):
    im_b = depth.shape[0]
    im_ch = depth.shape[1]
    im_h = depth.shape[2]
    im_w = depth.shape[3]

    depth_np = to_numpy_image(depth)

    images = torch.zeros((im_b, 3, im_h, im_w), dtype=torch.uint8, device=depth.device)
    for b in range(im_b):
        cur_depth = depth_np[b, :, :, 0]
        cur_depth = np.clip(cur_depth / max_depth * 255, 0, 255).astype(np.uint8)
        cur_depth_img = cv2.applyColorMap(255 - cur_depth, colormap=cv2.COLORMAP_JET)
        images[b: b+1, :, :, :] = to_torch_image(cur_depth_img[None, :, :, :], depth.device)
    return images


def draw_normal_images(normal):
    return torch.clamp((normal + 1.0) / 2.0 * 255.0, min=0, max=255).type(torch.uint8)


# def draw_segmentation_images(image, boxes, masks):
#     im_b = image.shape[0]
#     im_ch = image.shape[1]
#     im_h = image.shape[2]
#     im_w = image.shape[3]
#
#     box_image = image.copy()
#
#     segmentation_image = image * 0.0
#     for box, mask in zip(boxes, masks):
#         box = np.round(box).astype(np.int32)
#         mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]))
#         segmentation_image[box[0]:box[2], box[1]:box[3]] = np.minimum(
#                 segmentation_image[box[0]:box[2], box[1]:box[3]] + np.expand_dims(mask,
#                                                                                   axis=-1) * np.random.randint(
#                         255, size=(3,), dtype=np.int32), 255)
#
#     return segmentation_image


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0.5,
                                  np.minimum(image[:, :, c] *
                                             (1 - alpha) + alpha * color[c], 255),
                                  image[:, :, c])


def draw_instance_images(config, image, boxes, masks, class_ids, draw_mask=True):
    im_b = image.shape[0]
    im_ch = image.shape[1]
    im_h = image.shape[2]
    im_w = image.shape[3]

    masked_image = to_numpy_image(image).astype(np.uint8)

    for b in range(im_b):
        cur_boxes = boxes[b, :, :]
        cur_masks = masks[b, :, ...]
        cur_class_ids = class_ids[b, :]
        cur_masked_image = masked_image[b, :, :, :].copy()

        ## Number of instances
        N = len(cur_boxes)
        if not N:
            pass
        else:
            assert cur_boxes.shape[0] == cur_masks.shape[0] == cur_class_ids.shape[0]

        ## Generate random colors
        instance_colors = ColorPalette(N).getColorMap(returnTuples=True)

        class_colors = ColorPalette(11).getColorMap(returnTuples=True)
        class_colors[0] = (128, 128, 128)

        for i in range(N):
            ## Bounding box
            if not (cur_boxes[i, :].long() > 0).any():
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = cur_boxes[i, :].cpu().detach().numpy()

            ## Label
            class_id = int(cur_class_ids[i])

            if class_id > 0:
                ## Mask
                if len(cur_masks.shape) == 4:
                    mask = cur_masks[i, class_id, :, :]
                elif len(cur_masks.shape) == 3:
                    mask = cur_masks[i, :, :]
                else:
                    raise Exception('Unsupported mask shape')
                mask_full = resize_mask_full(config, cur_boxes[i, :].long(), mask)
                mask_full = mask_full.cpu().detach().numpy()
                apply_mask(cur_masked_image, mask_full, instance_colors[i])

                # cv2.line(cur_masked_image, (0, 0), (511, 511), (255, 0, 0), 5)
                #
                # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
                # pts = pts.reshape((-1, 1, 2))
                # cv2.polylines(cur_masked_image, [pts], True, (0, 255, 255))
                ## Mask Polygon
                ## Pad to ensure proper polygons for masks that touch image edges.
                if draw_mask:
                    padded_mask = np.zeros((mask_full.shape[0] + 2, mask_full.shape[1] + 2), dtype=mask_full.dtype)
                    padded_mask[1:-1, 1:-1] = mask_full
                    contours = find_contours(padded_mask, 0.5)
                    for verts in contours:
                        ## Subtract the padding and flip (y, x) to (x, y)
                        verts = np.fliplr(verts) - 1
                        cv2.polylines(cur_masked_image, [np.expand_dims(verts.astype(np.int32), 1)], True,
                                      color=class_colors[class_id])

        masked_image[b, :, :, :] = cur_masked_image

    masked_image = to_torch_image(masked_image, image.device)

    return masked_image


def detections_to_normal(config, image, boxes, masks, class_ids, plane_params):
    im_b = image.shape[0]
    im_ch = image.shape[1]
    im_h = image.shape[2]
    im_w = image.shape[3]
    dev = image.device

    normals = torch.zeros((im_b, 3, im_h, im_w), device=dev)
    for b in range(im_b):
        cur_boxes = boxes[b, :, :]
        cur_masks = masks[b, :, ...]
        cur_class_ids = class_ids[b, :]
        cur_plane_params = plane_params[b, :]

        ## Number of instances
        N = len(cur_boxes)
        if not N:
            pass
        else:
            assert cur_boxes.shape[0] == cur_masks.shape[0] == cur_class_ids.shape[0] == cur_plane_params.shape[0]

        for i in range(N):
            ## Bounding box
            if not (cur_boxes[i, :].long() > 0).any():
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = cur_boxes[i, :].cpu().detach().numpy()

            ## Label
            class_id = int(cur_class_ids[i])

            if class_id > 0:
                ## Mask
                if len(cur_masks.shape) == 4:
                    mask = cur_masks[i, class_id, :, :]
                elif len(cur_masks.shape) == 3:
                    mask = cur_masks[i, :, :]
                else:
                    raise Exception('Unsupported mask shape')
                mask_full = resize_mask_full(config, cur_boxes[i, :].long(), mask)

                plane_normal = F.normalize(cur_plane_params[i, :], dim=0)
                normals[b, :, mask_full > 0.5] = plane_normal.view(-1, 1)
    return normals


## Fit a 3D plane from points
def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]
    return


## Fit a 3D plane from points
def fitPlaneSVD(points):
    centroid = points.mean(axis=0, keepdims=True)
    points_demean = points - centroid
    [U, S, V] = np.linalg.svd(points_demean, full_matrices=False)
    # for the smallest singular value
    normal = V[2, :]
    d = centroid[0, :].dot(normal)
    plane_eq = np.concatenate([normal, [-d]])

    return plane_eq


def get_ranges(camera):
    fx = camera[0]
    fy = camera[1]
    cx = camera[2]
    cy = camera[3]
    w = camera[4]
    h = camera[5]

    urange = (np.arange(w, dtype=np.float32) - cx) / fx
    urange = urange.reshape(1, -1).repeat(h, 0)
    vrange = (np.arange(h, dtype=np.float32) - cy) / fy
    vrange = vrange.reshape(-1, 1).repeat(w, 1)
    ones = np.ones_like(urange)

    # ranges = np.stack([urange, ones, -vrange], axis=2)
    ranges = np.stack([urange, vrange, ones], axis=2)

    return ranges


def get_ranges_torch_batch(camera, dev):
    fx = camera[:, 0]
    fy = camera[:, 1]
    cx = camera[:, 2]
    cy = camera[:, 3]
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    bsize = camera.shape[0]

    urange = (torch.arange(w, dtype=camera.dtype, device=dev).view(1, -1) - cx.view(bsize, 1)) / fx.view(bsize, 1)
    urange = urange.view(bsize, 1, 1, -1).repeat(1, 1, h, 1)
    vrange = (torch.arange(h, dtype=camera.dtype, device=dev).view(1, -1) - cy.view(bsize, 1)) / fy.view(bsize, 1)
    vrange = vrange.view(bsize, 1, -1, 1).repeat(1, 1, 1, w)
    ones = torch.ones_like(urange)

    # ranges = np.stack([urange, ones, -vrange], axis=2)
    ranges = torch.cat([urange, vrange, ones], dim=1)

    return ranges


def get_ranges_torch(camera, dev):
    fx = camera[0]
    fy = camera[1]
    cx = camera[2]
    cy = camera[3]
    # all images in a batch should have the same dimensions
    w = int(camera[4])
    h = int(camera[5])

    urange = (torch.arange(w, dtype=camera.dtype, device=dev) - cx) / fx
    urange = urange.view(1, 1, 1, -1).repeat(1, 1, h, 1)
    vrange = (torch.arange(h, dtype=camera.dtype, device=dev) - cy) / fy
    vrange = vrange.view(1, 1, -1, 1).repeat(1, 1, 1, w)
    ones = torch.ones_like(urange)

    # ranges = np.stack([urange, ones, -vrange], axis=2)
    ranges = torch.cat([urange, vrange, ones], dim=1)

    return ranges


def get_ranges_pad_torch_batch(camera):
    fx = camera[:, 0]
    fy = camera[:, 1]
    cx = camera[:, 2]
    cy = camera[:, 3]
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    dcy = (w - h) / 2
    bsize = camera.shape[0]

    dev = camera.device
    dtype = camera.dtype

    urange = (torch.arange(w, device=dev, dtype=dtype).view(1, -1) - cx.view(bsize, 1)) / fx.view(bsize, 1)
    urange = urange.view(bsize, 1, 1, -1).repeat(1, 1, w, 1)
    vrange = (torch.arange(w, device=dev, dtype=dtype).view(1, -1) - cy.view(bsize, 1) - dcy) / fy.view(bsize, 1)
    vrange = vrange.view(bsize, 1, -1, 1).repeat(1, 1, 1, w)
    ones = torch.ones_like(urange)

    # ranges = np.stack([urange, ones, -vrange], axis=2)
    ranges = torch.cat([urange, vrange, ones], dim=1)

    return ranges


def get_ranges_pad_torch(camera):
    fx = camera[0]
    fy = camera[1]
    cx = camera[2]
    cy = camera[3]
    # all images in a batch should have the same dimensions
    w = int(camera[4])
    h = int(camera[5])
    dcy = (w - h) / 2

    dev = camera.device
    dtype = camera.dtype

    urange = (torch.arange(w, device=dev, dtype=dtype) - cx) / fx
    urange = urange.view(1, 1, 1, -1).repeat(1, 1, w, 1)
    vrange = (torch.arange(w, device=dev, dtype=dtype) - cy - dcy) / fy
    vrange = vrange.view(1, 1, -1, 1).repeat(1, 1, 1, w)
    ones = torch.ones_like(urange)

    # ranges = np.stack([urange, ones, -vrange], axis=2)
    ranges = torch.cat([urange, vrange, ones], dim=1)

    return ranges


def get_coords_pad_torch_batch(camera, max_disp, min_disp=5, factor=4):
    fx = camera[:, 0].view(-1, 1, 1, 1, 1)
    fy = camera[:, 1].view(-1, 1, 1, 1, 1)
    cx = camera[:, 2].view(-1, 1, 1, 1, 1)
    cy = camera[:, 3].view(-1, 1, 1, 1, 1)
    # all images in a batch should have the same dimensions
    im_w = int(camera[0, 4])
    im_h = int(camera[0, 5])
    im_w_f = im_w // factor
    im_h_f = im_h // factor
    max_disp_f = max_disp // factor
    b = camera[:, 6].view(-1, 1, 1, 1, 1)
    dcy = (im_w - im_h) / 2
    bsize = camera.shape[0]

    dev = camera.device

    us = torch.arange(0, im_w, step=factor, device=dev, dtype=camera.dtype).view(1, 1, 1, 1, im_w_f) - cx
    vs = torch.arange(0, im_w, step=factor, device=dev, dtype=camera.dtype).view(1, 1, 1, im_w_f, 1) - (cy + dcy)
    ds = torch.arange(0, max_disp, step=factor, device=dev, dtype=camera.dtype).view(1, 1, max_disp_f, 1, 1)
    pts_uvd = torch.cat([us.repeat(1, 1, max_disp_f, im_w_f, 1),
                         vs.repeat(1, 1, max_disp_f, 1, im_w_f),
                         ds.repeat(1, 1, 1, im_w_f, im_w_f)], dim=1)

    pts_xyz = pts_uvd.new_empty(pts_uvd.shape)
    pts_xyz[:, 0:1, :, :, :] = pts_uvd[:, 0:1, :, :, :] * fx * b / torch.clamp(pts_uvd[:, 2:3, :, :, :], min=min_disp) / fx
    pts_xyz[:, 1:2, :, :, :] = pts_uvd[:, 1:2, :, :, :] * fx * b / torch.clamp(pts_uvd[:, 2:3, :, :, :], min=min_disp) / fy
    pts_xyz[:, 2:3, :, :, :] = fx * b / torch.clamp(pts_uvd[:, 2:3, :, :, :], min=min_disp)

    return pts_xyz


def get_coords_uvd_pad_torch_batch(camera, config, max_disp, min_disp=5, factor=4, px=0, py=0):
    # all images in a batch should have the same dimensions
    im_w = int(camera[0, 4])
    im_h = int(camera[0, 5])
    im_w_f = im_w // factor
    im_h_f = im_h // factor
    # px_f = px // factor
    # py_f = py // factor
    max_disp_f = max_disp // factor
    a = config.UVD_CONST
    bsize = camera.shape[0]

    dev = camera.device

    us = torch.arange(0, im_w, step=factor, device=dev, dtype=camera.dtype).view(1, 1, 1, 1, im_w_f) - px
    vs = torch.arange(0, im_w, step=factor, device=dev, dtype=camera.dtype).view(1, 1, 1, im_w_f, 1) - py
    ds = torch.arange(0, max_disp, step=factor, device=dev, dtype=camera.dtype).view(1, 1, max_disp_f, 1, 1)
    pts_uvdun = torch.cat([us.repeat(bsize, 1, max_disp_f, im_w_f, 1),
                           vs.repeat(bsize, 1, max_disp_f, 1, im_w_f),
                           ds.repeat(bsize, 1, 1, im_w_f, im_w_f)], dim=1)

    pts_uvd = pts_uvdun.new_empty(pts_uvdun.shape)
    pts_uvd[:, 0:1, :, :, :] = pts_uvdun[:, 0:1, :, :, :] / torch.clamp(pts_uvdun[:, 2:3, :, :, :], min=min_disp)
    pts_uvd[:, 1:2, :, :, :] = pts_uvdun[:, 1:2, :, :, :] / torch.clamp(pts_uvdun[:, 2:3, :, :, :], min=min_disp)
    pts_uvd[:, 2:3, :, :, :] = a / torch.clamp(pts_uvd[:, 2:3, :, :, :], min=min_disp)

    return pts_uvd


def resize_mask_full(config, box, mask):
    dev = mask.device
    dtype = mask.dtype

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.upsample(mask, size=(int(box[2] - box[0]), int(box[3] - box[1])), mode='bilinear')
    mask = mask.squeeze(0).squeeze(0)

    final_mask = torch.zeros(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, device=dev, dtype=dtype)
    final_mask[box[0]:box[2], box[1]:box[3]] = mask

    return final_mask


def resize_mask_relative(im_h, im_w, box, mask):
    dev = mask.device
    dtype = mask.dtype

    box_pix = box.clone()
    box_pix[[0, 2]] *= im_h
    box_pix[[1, 3]] *= im_w
    box_pix = box_pix.long()

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.upsample(mask, size=(box_pix[2] - box_pix[0], box_pix[3] - box_pix[1]), mode='bilinear')
    mask = mask.squeeze(0).squeeze(0)

    final_mask = torch.zeros(im_h, im_w, device=dev, dtype=dtype)
    final_mask[box_pix[0]:box_pix[2], box_pix[1]:box_pix[3]] = mask

    return final_mask


def points_to_plane(points, plane):
    points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=-1)
    diff = points @ plane

    return diff


def fit_plane_torch(points):
    if points.shape[0] == points.shape[1]:
        return torch.solve(torch.ones(points.shape[0], 1, device=points.device), points)[0]
    else:
        return torch.lstsq(torch.ones(points.shape[0], 1, device=points.device), points)[0][0:3, 0]
    return


def fit_plane_ransac(points):
    num_iter = 10
    plane_diff_threshold = 0.01

    best_inliers = 0
    best_inliers_mask = None
    idxs = np.arange(0, points.shape[0])
    for i in range(num_iter):
        cur_idxs = np.random.choice(idxs, 3)
        cur_points = points[cur_idxs]
        try:
            cur_plane = fitPlane(cur_points)
        except:
            continue

        # relative distance to the plane
        diff = np.abs(np.matmul(points, cur_plane) - np.ones(points.shape[0]))
        inlier_mask = diff < plane_diff_threshold

        cur_inliers = inlier_mask.sum()
        if cur_inliers > best_inliers:
            best_inliers = cur_inliers
            best_inliers_mask = inlier_mask

        # if enough inliers
        if cur_inliers / points.shape[0] > 0.95:
            break

    best_plane = fitPlane(points[best_inliers_mask])

    return best_inliers_mask, best_plane


def fit_plane_ransac_torch(points, plane_diff_threshold=0.01, absolute=False):
    num_iter = 50

    best_inliers = 0
    best_inliers_mask = None
    for i in range(num_iter):
        cur_idxs = torch.randperm(points.shape[0])[:3]
        cur_points = points[cur_idxs]
        try:
            cur_plane = fit_plane_torch(cur_points)
        except:
            continue

        # relative distance to the plane
        diff = torch.abs(torch.matmul(points, cur_plane).view(-1) - torch.ones(points.shape[0], device=points.device))
        if absolute:
            diff = diff / cur_plane.norm()
        inlier_mask = diff < plane_diff_threshold

        cur_inliers = inlier_mask.sum()
        if cur_inliers > best_inliers:
            best_inliers = cur_inliers
            best_inliers_mask = inlier_mask

        # if enough inliers
        if cur_inliers.float() / points.shape[0] > 0.95:
            break

    if best_inliers < 3:
        print('best_inliers', best_inliers)
        best_plane = torch.zeros(3, dtype=torch.float, device=points.device)
    else:
        best_plane = fit_plane_torch(points[best_inliers_mask])

    return best_inliers_mask, best_plane


def fit_plane_dist_ransac_torch(points, normal, plane_diff_threshold=0.01, absolute=False):
    num_iter = 50
    normal = normal / normal.norm()

    best_inliers = 0
    best_inliers_mask = torch.zeros((points.shape[0]), dtype=torch.bool, device=points.device)
    for i in range(num_iter):
        cur_idx = torch.randint(points.shape[0], (1,))[0]
        cur_point = points[cur_idx]

        cur_d = cur_point.dot(normal)
        cur_plane = normal / cur_d

        # relative distance to the plane
        diff = torch.abs(torch.matmul(points, cur_plane).view(-1) - torch.ones(points.shape[0], device=points.device))
        if absolute:
            diff = diff / cur_plane.norm()
        inlier_mask = diff < plane_diff_threshold

        cur_inliers = inlier_mask.sum()
        if cur_inliers > best_inliers:
            best_inliers = cur_inliers
            best_inliers_mask = inlier_mask

        # if enough inliers
        if cur_inliers.float() / points.shape[0] > 0.9:
            break

    if best_inliers < 3:
        print('best_inliers', best_inliers)

    best_points = points[best_inliers_mask]
    best_d = best_points.matmul(normal.view(3, 1)).mean()
    best_plane = normal / best_d

    return best_inliers_mask, best_plane


def calc_points_depth(depth, K):
    h = depth.shape[0]
    w = depth.shape[1]
    urange = np.arange(w, dtype=np.float32).reshape(1, -1).repeat(h, 0)
    vrange = np.arange(h, dtype=np.float32).reshape(-1, 1).repeat(w, 1)
    ranges = np.stack([urange, vrange, np.ones(urange.shape)], axis=-1)

    points = np.matmul(np.linalg.inv(K), np.expand_dims(ranges, axis=-1)).squeeze(axis=-1)
    points *= np.expand_dims(depth, axis=-1)

    return points


## Clean segmentation
def cleanSegmentation(image,
                      planes,
                      segmentation,
                      depth,
                      camera,
                      planeAreaThreshold=200,
                      planeWidthThreshold=10,
                      depthDiffThreshold=0.1,
                      validAreaThreshold=0.5,
                      brightThreshold=20,
                      return_plane_depths=False):

    planeDepths = calcPlaneDepths(planes, segmentation.shape[1], segmentation.shape[0], camera).transpose((2, 0, 1))

    newSegmentation = np.full(segmentation.shape, fill_value=-1)
    validMask = np.logical_and(np.linalg.norm(image, axis=-1) > brightThreshold, depth > 1e-4)
    depthDiffMask = np.logical_or(np.abs(planeDepths - depth) < depthDiffThreshold, depth < 1e-4)
    # depthDiffMask = np.logical_or(np.abs(planeDepths - depth) < 1.0, depth < 1e-4)

    for segmentIndex in np.unique(segmentation):
        if segmentIndex < 0:
            continue
        segmentMask = segmentation == segmentIndex

        oriArea = segmentMask.sum()
        segmentMask = np.logical_and(segmentMask, depthDiffMask[segmentIndex])

        newArea = np.logical_and(segmentMask, validMask).sum()
        if newArea < oriArea * validAreaThreshold:
            continue
        segmentMask = segmentMask.astype(np.uint8)
        segmentMask = cv2.dilate(segmentMask, np.ones((3, 3)))
        numLabels, components = cv2.connectedComponents(segmentMask)
        for label in range(1, numLabels):
            mask = components == label
            ys, xs = mask.nonzero()
            area = float(len(xs))
            if area < planeAreaThreshold:
                continue
            size_y = ys.max() - ys.min() + 1
            size_x = xs.max() - xs.min() + 1
            length = np.linalg.norm([size_x, size_y])
            if area / length < planeWidthThreshold:
                continue
            newSegmentation[mask] = segmentIndex
            continue
        continue
    if return_plane_depths:
        return newSegmentation, planeDepths
    return newSegmentation


def roi_align(input, boxes, output_size, sampling_ratio=-1):
    h = input.shape[2]
    w = input.shape[3]
    boxes_pt = []
    for box in boxes:
        y1, x1, y2, x2 = box.chunk(4, dim=1)
        # convert to unnormalized coordinates in [x1, y1, x2, y2] format
        box_pt = torch.cat([x1 * (w - 1), y1 * (h - 1), x2 * (w - 1), y2 * (h - 1)], dim=1)
        boxes_pt.append(box_pt)
    return torchvision.ops.roi_align(input, boxes_pt, output_size, sampling_ratio=sampling_ratio, aligned=True)


def roi_align_batch(input, boxes, output_size, sampling_ratio=-1):
    dev = input.device
    dtype = input.dtype

    ch = input.shape[1]
    h = input.shape[2]
    w = input.shape[3]

    n = boxes.shape[0]

    b, y1, x1, y2, x2 = boxes.chunk(5, dim=1)
    # convert to unnormalized coordinates in [x1, y1, x2, y2] format
    boxes_pt = torch.cat([b, x1 * (w - 1), y1 * (h - 1), x2 * (w - 1), y2 * (h - 1)], dim=1)
    valid_idxs = torchvision.ops.box_area(boxes_pt[:, 1:5]) > 0

    pooled = torch.zeros((n, ch, output_size[0], output_size[1]), device=dev, dtype=dtype)
    pooled[valid_idxs, :, :, :] = torchvision.ops.roi_align(input, boxes_pt[valid_idxs, :], output_size, sampling_ratio=sampling_ratio, aligned=True)

    return pooled


def plane_to_plane_dist(plane1, plane2):
    plane1_d = 1.0 / torch.norm(plane1)
    p1 = plane1 * plane1_d * plane1_d
    # d = (p1.dot(plane2) - 1.0) / plane2.norm()
    d = (p1.dot(plane2) - 1.0)

    return d


def plane_to_plane_dot(plane1, plane2):
    return plane1.dot(plane2) / (torch.norm(plane1) * torch.norm(plane2))


def calc_iou(box1, box2):
    y1_i = max(box1[0], box2[0])
    y2_i = min(box1[2], box2[2])
    x1_i = max(box1[1], box2[1])
    x2_i = min(box1[3], box2[3])

    area_i = max(y2_i - y1_i, 0.0) * max(x2_i - x1_i, 0.0)
    area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # iou = area_i / (area_1 + area_2 - area_i)
    iou = area_i / area_2

    if iou > 1:
        print('iou = ', iou)

    return iou


def calc_iou_batch(boxes):
    n = boxes.shape[0]
    x1 = boxes[:, 1]
    x2 = boxes[:, 3]
    y1 = boxes[:, 0]
    y2 = boxes[:, 2]

    x1_cols = x1[:, None].repeat(1, n)
    x1_rows = x1[None, :].repeat(n, 1)
    x2_cols = x2[:, None].repeat(1, n)
    x2_rows = x2[None, :].repeat(n, 1)
    y1_cols = y1[:, None].repeat(1, n)
    y1_rows = y1[None, :].repeat(n, 1)
    y2_cols = y2[:, None].repeat(1, n)
    y2_rows = y2[None, :].repeat(n, 1)

    x1_i = torch.max(x1_cols, x1_rows)
    x2_i = torch.min(x2_cols, x2_rows)
    y1_i = torch.max(y1_cols, y1_rows)
    y2_i = torch.min(y2_cols, y2_rows)

    area_i = torch.clamp(y2_i - y1_i, min=0.0) * torch.clamp(x2_i - x1_i, min=0.0)
    area = (y2 - y1) * (x2 - x1)

    iou = area_i / area

    return iou


def rotx(ang):
    R = torch.zeros([ang.shape[0], 3, 3], dtype=torch.float, device=ang.device)
    sin_ang = torch.sin(ang)
    cos_ang = torch.cos(ang)
    R[:, 0, 0] = 1.0
    R[:, 1, 1] = cos_ang
    R[:, 1, 2] = -sin_ang
    R[:, 2, 1] = sin_ang
    R[:, 2, 2] = cos_ang

    return R


def roty(ang):
    R = torch.zeros([ang.shape[0], 3, 3], dtype=torch.float, device=ang.device)
    sin_ang = torch.sin(ang)
    cos_ang = torch.cos(ang)
    R[:, 0, 0] = cos_ang
    R[:, 0, 2] = sin_ang
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -sin_ang
    R[:, 2, 2] = cos_ang

    return R


def sinc(ang, eps=1.0e-6):
    vals = ang.new_empty(ang.shape)

    small_angle_idxs = ang.abs() < eps
    large_angle_idxs = small_angle_idxs.logical_not()
    if small_angle_idxs.sum() > 0:
        # Taylor expansion
        ang_sq = ang[small_angle_idxs].square()
        vals[small_angle_idxs] = 1.0 - ang_sq/6.0 + ang_sq.square()/120.0

    if large_angle_idxs.sum() > 0:
        vals[large_angle_idxs] = torch.sin(ang[large_angle_idxs]) / ang[large_angle_idxs]

    return vals


def ang_to_normal(ang, towards=True, eps=1.0e-6):
    if towards:
        a = 1
    else:
        a = -1
    normal = ang.new_empty([ang.shape[0], 3])
    th = ang.norm(dim=1)
    sinc_th = sinc(th, eps)

    normal[:, 0] = -a * ang[:, 1] * sinc_th
    normal[:, 1] = a * ang[:, 0] * sinc_th
    normal[:, 2] = -a * torch.cos(th)

    return normal


def ang_to_normal_im(ang, towards=True, eps=1.0e-6):
    if towards:
        a = 1
    else:
        a = -1
    normal = ang.new_empty([ang.shape[0], 3, ang.shape[2], ang.shape[3]])
    th = ang.norm(dim=1, keepdim=True)
    sinc_th = sinc(th, eps)

    normal[:, 0:1, :, :] = -a * ang[:, 1:2, :, :] * sinc_th
    normal[:, 1:2, :, :] = a * ang[:, 0:1, :, :] * sinc_th
    normal[:, 2:3, :, :] = -a * torch.cos(th)

    return normal


def normal_to_ang(normal, towards=True, eps=1.0e-6):
    if towards:
        a = 1
    else:
        a = -1
    ang = normal.new_empty((normal.shape[0], 2))

    th = torch.acos(-a * normal[:, 2].clamp(min=-1.0 + eps, max=1.0 - eps))
    sinc_th = sinc(th, eps)

    ang[:, 0] = a * normal[:, 1] / sinc_th
    ang[:, 1] = -a * normal[:, 0] / sinc_th

    return ang


def normal_to_ang_im(normal, towards=True, eps=1.0e-6):
    if towards:
        a = 1
    else:
        a = -1
    ang = normal.new_empty((normal.shape[0], 2, normal.shape[2], normal.shape[3]))

    # avoid computing acos in -1 and 1, because grad is nan
    th = torch.acos(-a * normal[:, 2:3, :, :].clamp(min=-1.0 + eps, max=1.0 - eps))
    sinc_th = sinc(th, eps)

    ang[:, 0:1, :, :] = a * normal[:, 1:2, :, :] / sinc_th
    ang[:, 1:2, :, :] = -a * normal[:, 0:1, :, :] / sinc_th

    return ang


def ang_to_class(camera, config, ang):
    vert_class = torch.clamp((ang[:, 0] + config.ANG_VERT / 2.0 + math.pi / 2.0) // config.ANG_VERT_RES, min=0, max=config.ANG_BINS - 1)
    hor_class = torch.clamp((ang[:, 1] + config.ANG_HOR/2.0 + math.pi / 2.0) // config.ANG_HOR_RES, min=0, max=config.ANG_BINS - 1)

    ang_class = torch.stack([vert_class, hor_class], dim=1)

    return ang_class


def class_to_ang(camera, config, ang_class):
    vert_ang = - config.ANG_VERT / 2.0 - math.pi / 2.0 + ang_class[:, 0] * config.ANG_VERT_RES
    hor_ang = - config.ANG_HOR / 2.0 - math.pi / 2.0 + ang_class[:, 1] * config.ANG_HOR_RES

    ang = torch.stack([vert_ang, hor_ang], dim=1)

    return ang


def ang_to_class_cont(camera, config, ang):
    vert_class = (ang[:, 0] + config.ANG_VERT / 2.0 + math.pi / 2.0) / config.ANG_VERT_RES
    hor_class = (ang[:, 1] + config.ANG_HOR/2.0 + math.pi / 2.0) / config.ANG_HOR_RES

    ang_class = torch.stack([vert_class, hor_class], dim=1)

    return ang_class


def normal_uvd_to_normal(camera, config, normal_uvd, px=0, py=0):
    n_dims = len(normal_uvd.shape)
    view_dims = [1 for i in range(n_dims)]
    view_dims[0] = -1

    fx = camera[:, 0].view(*view_dims)
    fy = camera[:, 1].view(*view_dims)
    cx = camera[:, 2].view(*view_dims)
    cy = camera[:, 3].view(*view_dims)
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    dcy = (w - h) / 2
    b = camera[:, 6].view(*view_dims)
    a = config.UVD_CONST

    normal = normal_uvd.new_empty(normal_uvd.shape)
    normal[..., 0:1] = normal_uvd[..., 0:1] / b
    normal[..., 1:2] = normal_uvd[..., 1:2] * fy / (fx * b)
    normal[..., 2:3] = (normal_uvd[..., 0:1] * (cx - px) +
                        normal_uvd[..., 1:2] * (cy + dcy - py) +
                        normal_uvd[..., 2:3] * a) / (fx * b)

    return normal


def normal_uvd_to_normal_im(camera, config, normal_uvd, px=0, py=0):
    im_b = normal_uvd.shape[0]

    fx = camera[:, 0].view(im_b, 1, 1, 1)
    fy = camera[:, 1].view(im_b, 1, 1, 1)
    cx = camera[:, 2].view(im_b, 1, 1, 1)
    cy = camera[:, 3].view(im_b, 1, 1, 1)
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    dcy = (w - h) / 2
    b = camera[:, 6].view(im_b, 1, 1, 1)
    a = config.UVD_CONST

    normal = normal_uvd.new_empty(normal_uvd.shape)
    normal[:, 0:1, :, :] = normal_uvd[:, 0:1, :, :] / b
    normal[:, 1:2, :, :] = normal_uvd[:, 1:2, :, :] * fy / (fx * b)
    normal[:, 2:3, :, :] = (normal_uvd[:, 0:1, :, :] * (cx - px) +
                            normal_uvd[:, 1:2, :, :] * (cy + dcy - py) +
                            normal_uvd[:, 2:3, :, :] * a) / (fx * b)

    return normal


def normal_to_normal_uvd(camera, config, normal, px=0, py=0):
    n_dims = len(normal.shape)
    view_dims = [1 for i in range(n_dims)]
    view_dims[0] = -1

    fx = camera[:, 0].view(*view_dims)
    fy = camera[:, 1].view(*view_dims)
    cx = camera[:, 2].view(*view_dims)
    cy = camera[:, 3].view(*view_dims)
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    dcy = (w - h) / 2
    b = camera[:, 6].view(*view_dims)
    a = config.UVD_CONST

    normal_uvd = normal.new_empty(normal.shape)
    normal_uvd[..., 0:1] = normal[..., 0:1] * b
    normal_uvd[..., 1:2] = normal[..., 1:2] * (fx * b) / fy
    normal_uvd[..., 2:3] = (-normal[..., 0:1] * b * (cx - px) +
                            -normal[..., 1:2] * fx * b * (cy + dcy - py) / fy +
                            normal[..., 2:3] * fx * b) / a

    return normal_uvd


def normal_to_normal_uvd_im(camera, config, normal, px=0, py=0):
    im_b = normal.shape[0]

    fx = camera[:, 0].view(im_b, 1, 1, 1)
    fy = camera[:, 1].view(im_b, 1, 1, 1)
    cx = camera[:, 2].view(im_b, 1, 1, 1)
    cy = camera[:, 3].view(im_b, 1, 1, 1)
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    dcy = (w - h) / 2
    b = camera[:, 6].view(im_b, 1, 1, 1)
    a = config.UVD_CONST

    normal_uvd = normal.new_empty(normal.shape)
    normal_uvd[:, 0:1, :, :] = normal[:, 0:1, :, :] * b
    normal_uvd[:, 1:2, :, :] = normal[:, 1:2, :, :] * (fx * b) / fy
    normal_uvd[:, 2:3, :, :] = (-normal[:, 0:1, :, :] * b * (cx - px) +
                                -normal[:, 1:2, :, :] * fx * b * (cy + dcy - py) / fy +
                                normal[:, 2:3, :, :] * fx * b) / a

    return normal_uvd


def normal_to_ang_uvd_im(camera, config, normal):
    im_b = normal.shape[0]
    im_h = normal.shape[2]
    im_w = normal.shape[3]

    normal_uvd = normal_to_normal_uvd_im(camera, config, normal)
    normal_uvd = normal_uvd.view(3, -1)

    ang_uvd = normal_to_ang(normal_uvd.permute(1, 0)).permute(1, 0)
    ang_uvd = ang_uvd.view(im_b, 2, im_h, im_w)

    return ang_uvd


def ang_uvd_to_normal_im(camera, config, ang_uvd):
    im_b = ang_uvd.shape[0]
    im_h = ang_uvd.shape[2]
    im_w = ang_uvd.shape[3]

    assert im_b == 1

    ang_uvd = ang_uvd.view(2, -1)
    normal_uvd = ang_to_normal(ang_uvd.permute(1, 0)).permute(1, 0)
    normal_uvd = normal_uvd.view(im_b, 3, im_h, im_w)

    normal = normal_uvd_to_normal_im(camera, config, normal_uvd)

    return normal


def normal_uvdun_to_normal_im(camera, normal_uvd, disp):
    fx = camera[:, 0]
    fy = camera[:, 1]
    cx = camera[:, 2]
    cy = camera[:, 3]
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    dcy = (w - h) / 2

    dev = normal_uvd.device
    dtype = normal_uvd.dtype

    im_b = normal_uvd.shape[0]
    im_h = normal_uvd.shape[2]
    im_w = normal_uvd.shape[3]

    us = torch.arange(w, device=dev, dtype=dtype).view(1, -1) - cx.view(im_b, 1)
    vs = torch.arange(w, device=dev, dtype=dtype).view(1, -1) - cy.view(im_b, 1) - dcy
    us = us.view(im_b, 1, 1, im_w).repeat(1, 1, im_h, 1)
    vs = vs.view(im_b, 1, im_h, 1).repeat(1, 1, 1, im_w)

    normal = normal_uvd.new_empty((im_b, 3, im_h, im_w))
    normal[:, 0:1, :, :] = -normal_uvd[:, 0:1, :, :] * fx[:, None, None, None]
    normal[:, 1:2, :, :] = -normal_uvd[:, 1:2, :, :] * fy[:, None, None, None]
    normal[:, 2:3, :, :] = -(-normal_uvd[:, 0:1, :, :] * us + \
                          -normal_uvd[:, 1:2, :, :] * vs + \
                          disp)

    normal = F.normalize(normal, dim=1)

    return normal


def normal_to_normal_uvdun_im(camera, normal, disp):
    fx = camera[:, 0]
    fy = camera[:, 1]
    cx = camera[:, 2]
    cy = camera[:, 3]
    # all images in a batch should have the same dimensions
    w = int(camera[0, 4])
    h = int(camera[0, 5])
    dcy = (w - h) / 2
    eps = 1.0e-6

    dev = normal.device
    dtype = normal.dtype

    im_b = normal.shape[0]
    im_h = normal.shape[2]
    im_w = normal.shape[3]

    us = torch.arange(w, device=dev, dtype=dtype).view(1, -1) - cx.view(im_b, 1)
    vs = torch.arange(w, device=dev, dtype=dtype).view(1, -1) - cy.view(im_b, 1) - dcy
    us = us.view(im_b, 1, 1, im_w).repeat(1, 1, im_h, 1)
    vs = vs.view(im_b, 1, im_h, 1).repeat(1, 1, 1, im_w)

    denom = (normal[:, 0:1, :, :] * us / fx[:, None, None, None] +
             normal[:, 1:2, :, :] * vs / fy[:, None, None, None] +
             normal[:, 2:3, :, :])
    # clamp on both sides of 0
    denom = torch.where(torch.logical_and(denom.abs() < eps, denom < 0),
                        -torch.tensor([eps], device=dev, dtype=dtype),
                        denom)
    denom = torch.where(torch.logical_and(denom.abs() < eps, denom >= 0),
                        torch.tensor([eps], device=dev, dtype=dtype),
                        denom)
    normal_uvd = normal.new_empty((im_b, 2, im_h, im_w))
    normal_uvd[:, 0:1, :, :] = normal[:, 0:1, :, :] * disp / \
                             (fx[:, None, None, None] * denom)
    normal_uvd[:, 1:2, :, :] = normal[:, 1:2, :, :] * disp / \
                             (fy[:, None, None, None] * denom)

    return normal_uvd


def huber(vals, beta=1.0):
    ret_vals = torch.where(vals.abs() < beta, 0.5 * vals.square() / beta, vals.abs() - 0.5 * beta)

    return ret_vals


def calc_derivative(data):
    im_ch = data.shape[1]
    kernel_x = torch.tensor([[[[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]]]], dtype=data.dtype, device=data.device)
    kernel_y = torch.tensor([[[[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]]]], dtype=data.dtype, device=data.device)
    dx = F.conv2d(data, kernel_x, padding=1, groups=im_ch)
    dy = F.conv2d(data, kernel_y, padding=1, groups=im_ch)

    return dx, dy


def point_to_plane_depth_diff(plane, points):
    plane = plane / plane.norm().clamp(min=1.0e-5).square()
    # Assuming camera center is (0, 0, 0)
    points_depth = points[2:3, :]
    # points_range = points.norm(dim=0, keepdim=True)
    rays = points / points_depth.clamp(min=1.0e-5)
    plane_depths = 1.0 / (rays * plane.view(3, 1)).sum(dim=0)
    plane_depths = plane_depths.clamp(min=0.0, max=15.0)

    return points_depth - plane_depths


def point_to_plane_dist(plane, points):
    normal = plane / plane.norm().clamp(min=1.0e-5)
    points_rel = points - plane.view(3, 1)

    dists = (points_rel * normal.view(3, 1)).sum(dim=0)

    return dists


def evaluate_plane_dist_norm(config, camera, detections, masks, depth_gt, max_bins=6, depth_pred=None,
                             batch=None, profiler=PassThroughProfiler()):
    bsize = detections.shape[0]
    nsize = detections.shape[1]

    dev = depth_gt.device
    dtype = depth_gt.dtype

    boxes = detections[:, :, 0:4].long()
    class_ids = detections[:, :, 4].long()
    planes = detections[:, :, 6:9]

    ranges = get_ranges_pad_torch_batch(camera)
    bin_width = 50
    error_dist_hist = torch.zeros(max_bins, device=dev, dtype=dtype)
    error_norm_hist = torch.zeros(max_bins, device=dev, dtype=dtype)
    error_area_hist = torch.zeros(max_bins, device=dev, dtype=dtype)
    error_cnt_hist = torch.zeros(max_bins, device=dev, dtype=dtype)
    target_dist_hist = torch.zeros(max_bins, device=dev, dtype=dtype)

    XYZ = ranges * depth_gt
    valid_mask = depth_gt > 0.2

    XYZ_pred = ranges * depth_pred
    valid_mask_pred = depth_pred > 0.2

    for b in range(bsize):
        for n in range(nsize):
            if class_ids[b, n] > 0:
                full_mask = resize_mask_full(config, boxes[b, n, :], masks[b, n, class_ids[b, n], :, :])
                full_mask = torch.logical_and(full_mask > 0.5, valid_mask[b, :, :, :].squeeze(0))
                # gt points belonging to the plane
                XYZ_plane_gt = XYZ[b, :, full_mask]

                full_mask_pred = resize_mask_full(config, boxes[b, n, :], masks[b, n, class_ids[b, n], :, :])
                full_mask_pred = torch.logical_and(full_mask_pred > 0.5, valid_mask_pred[b, :, :, :].squeeze(0))
                # points belonging to the plane
                XYZ_plane_pred = XYZ_pred[b, :, full_mask_pred]

                area = XYZ_plane_gt.shape[1]
                plane_pred = planes[b, n, :]
                if area >= 500 and plane_pred.norm() > 1e-6:
                    inlier_mask_gt_np, plane_gt_np = utils_cpp_py.ransac_plane(XYZ_plane_gt.cpu().numpy(),
                                                                               50,
                                                                               0.02,
                                                                               False)
                    inlier_mask_gt = torch.from_numpy(inlier_mask_gt_np).type_as(planes)
                    plane_gt = torch.from_numpy(plane_gt_np).type_as(planes)
                    plane_gt = plane_gt / plane_gt.norm().clip(min=1.0e-5).square()

                    inliers_ratio = float(inlier_mask_gt.sum()) / XYZ_plane_gt.shape[1]

                    # if inliers_ratio <= 0.6:
                    #     inliers_full_mask = torch.zeros_like(full_mask)
                    #     inliers_full_mask[full_mask] = inlier_mask_gt
                    #     print('inliers ratio = %.3f' % inliers_ratio)

                    if plane_gt.norm() < 1.0e-5 or inliers_ratio < 0.6:
                        continue
                    normal_gt = plane_gt / plane_gt.norm()

                    # # RANSAC for comparison
                    # with profiler.profile('ransac'):
                    #     # inlier_mask_ransac, plane_ransac = fit_plane_ransac_torch(XYZ_plane_ransac.transpose(0, 1),
                    #     #                                                       plane_diff_threshold=0.05)
                    #     inlier_mask_ransac_np, plane_ransac_np = utils_cpp_py.ransac_plane(XYZ_plane_pred.cpu().numpy(),
                    #                                                                        200,
                    #                                                                        0.05,
                    #                                                                        False)
                    #     inlier_mask_ransac = torch.from_numpy(inlier_mask_ransac_np).type_as(planes)
                    #     plane_ransac = torch.from_numpy(plane_ransac_np).type_as(planes)
                    #     plane_ransac = plane_ransac / plane_ransac.norm().clip(min=1.0e-5).square()
                    #
                    # # inliers_ratio_ransac = float(inlier_mask_ransac.sum()) / XYZ_plane_ransac.shape[1]
                    # if plane_ransac.norm() < 1.0e-5:
                    #     continue
                    #
                    # # XYZ_plane_rel_ransac = XYZ_plane_gt - plane_ransac[:, None]
                    # normal_ransac = plane_ransac / plane_ransac.norm()
                    # # normal = target_params[m] / target_params[m].norm()
                    # # dot product with normal
                    # # cur_errors_dist_ransac = (XYZ_plane_rel_ransac * normal_ransac[:, None]).sum(dim=0).abs()
                    # cur_errors_dist_ransac = point_to_plane_dist(plane_ransac, XYZ_plane_gt[:, inlier_mask_gt > 0.5]).abs()
                    # cur_error_norm_ransac = torch.acos(torch.clamp(normal_ransac.dot(normal_gt), max=1.0)) * 180.0 / np.pi
                    #
                    # plane_pred = plane_ransac
                    # # --------------------------------------------

                    bin_idx = min(int(np.sqrt(area) // bin_width), max_bins - 1)

                    area_inliers = inlier_mask_gt.sum()
                    # XYZ_plane_rel = XYZ_plane_gt - plane_pred[:, None]
                    normal = plane_pred / plane_pred.norm()
                    # normal = target_params[m] / target_params[m].norm()
                    # dot product with normal
                    # cur_errors_dist = (XYZ_plane_rel * normal[:, None]).sum(dim=0).abs()
                    # cur_errors_dist = point_to_plane_dist(plane_pred, XYZ_plane_gt[:, inlier_mask_gt > 0.5]).abs()
                    cur_errors_dist = point_to_plane_depth_diff(plane_pred, XYZ_plane_gt[:, inlier_mask_gt > 0.5]).abs()

                    error_dist_hist[bin_idx] += cur_errors_dist.square().sum()

                    cur_error_norm = torch.acos(torch.clamp(normal.dot(normal_gt), max=1.0)) * 180.0 / np.pi
                    cur_error_norm = torch.min(cur_error_norm, 180.0 - cur_error_norm)

                    # count it pixel-wise
                    error_norm_hist[bin_idx] += cur_error_norm.square() * area_inliers

                    error_area_hist[bin_idx] += area_inliers
                    error_cnt_hist[bin_idx] += 1

                    target_dist_hist[bin_idx] += XYZ_plane_gt[:, inlier_mask_gt > 0.5].norm(dim=0).sum()

                    # if area > 10000 and cur_error_norm_ransac > cur_error_norm and \
                    #    cur_errors_dist_ransac.square().mean().sqrt() < 0.8 * cur_errors_dist.square().mean().sqrt():
                    # # if cur_errors_dist.square().mean().sqrt() > 10.0:
                    #     print(b, n)
                    #     print('rot %f vs %f' % (float(cur_error_norm_ransac), float(cur_error_norm)))
                    #     print('dist %f vs %f' % (float(cur_errors_dist_ransac.square().mean().sqrt()), float(cur_errors_dist.square().mean().sqrt())))
                    #     depth_rmse = point_to_plane_depth_diff(plane_pred, XYZ_plane_gt).square().mean().sqrt()
                    #     depth_rmse_ransac = point_to_plane_depth_diff(plane_ransac, XYZ_plane_gt).square().mean().sqrt()
                    #     print('depth %f vs %f' % (float(depth_rmse_ransac), float(depth_rmse)))
                    #     print(plane_gt)
                    #     print(plane_pred)
                    #     print(plane_ransac)
                    #     # inlier_mask_comp, plane_comp = fit_plane_dist_ransac_torch(XYZ_plane_pred.transpose(0, 1),
                    #     #                                                                normal,
                    #     #                                                                plane_diff_threshold=0.05)
                    #     # plane_comp = plane_comp / plane_comp.norm().clip(min=1.0e-5).square()
                    #     # print(plane_comp)
                    #     # cur_errors_dist_comp = point_to_plane_depth_diff(plane_pred, XYZ_plane_gt).abs()

    return [error_dist_hist, error_norm_hist, error_area_hist, error_cnt_hist, target_dist_hist]


def find_plane_ids(config,
                   det_rois,
                   det_masks,
                   det_class_ids,
                   gt_boxes,
                   gt_masks,
                   gt_class_ids,
                   gt_plane_ids):
    bsize = det_masks.shape[0]
    nsize = det_masks.shape[1]

    dev = det_masks.device
    dtype = det_masks.dtype

    det_plane_ids = -1 * torch.ones((bsize, nsize), dtype=torch.int32, device=dev)
    det_mask_areas = torch.zeros((bsize, nsize), dtype=dtype, device=dev)
    for b in range(bsize):
        nsize_gt = (gt_class_ids[b, :] > 0).sum()

        if nsize_gt > 0:
            gt_masks_full = torch.zeros((nsize_gt,
                                         config.IMAGE_MAX_DIM,
                                         config.IMAGE_MAX_DIM),
                                        dtype=dtype, device=dev)

            for n in range(nsize_gt):
                if gt_class_ids[b, n] > 0:
                    gt_masks_full[n, :, :] = utils.resize_mask_full(config, gt_boxes[b, n, :].long(), gt_masks[b, n, :, :])

            for n in range(nsize):
                class_id = det_class_ids[b, n].long()
                if class_id > 0:
                    cur_det_mask = torch.where(det_masks[b, n, class_id, :, :] > 0.5,
                                               torch.tensor([1.0], dtype=dtype, device=dev),
                                               torch.tensor([0.0], dtype=dtype, device=dev))
                    cur_det_mask_full = utils.resize_mask_full(config,
                                                               det_rois[b, n, :].long(),
                                                               cur_det_mask)
                    mask_area = cur_det_mask_full.sum()
                    det_mask_areas[b, n] = mask_area

                    mask_intersection = cur_det_mask_full.expand(nsize_gt, -1, -1) * gt_masks_full
                    mask_int_area = mask_intersection.sum(dim=(-2, -1))
                    best_idx = torch.argmax(mask_int_area)
                    if mask_int_area[best_idx] / mask_area.clamp(min=1.0) > 0.5:
                        det_plane_ids[b, n] = gt_plane_ids[b, best_idx]

    return det_plane_ids, det_mask_areas


def rgb2hsv(input, epsilon=1e-10):
    assert(input.shape[1] == 3)

    input = input / 255.0

    r, g, b = input[:, 0], input[:, 1], input[:, 2]
    max_rgb, argmax_rgb = input.max(1)
    min_rgb, argmin_rgb = input.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon) * 255.0
    v = max_rgb * 255.0

    return torch.stack((h, s, v), dim=1)
