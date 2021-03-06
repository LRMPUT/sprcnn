"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import Dataset

import numpy as np
import time
import os
import sys
import utils
from datasets.scenenet_rgbd_scene import ScenenetRgbdScene


class ScenenetRgbdDatasetSingle(Dataset):
    def __init__(self,
                 options,
                 config,
                 split,
                 random=False,
                 load_annotation=True,
                 load_normals=True,
                 load_scores=False,
                 filter_depth=True,
                 annotation_dir='annotation',
                 writer=None):
        self.options = options
        self.config = config
        self.split = split
        self.random = random
        self.load_annotation = load_annotation
        self.load_normals = load_normals
        self.load_scores = load_scores
        self.filter_depth = filter_depth
        
        self.dataFolder = options.dataFolder

        self.writer = writer
        
        self.scenes = []
        self.sceneImageIndices = []

        self.scene_id_sc_plane_to_plane_id = {}

        scene_id_to_idx = {}
        next_plane_id = 0
        with open(os.path.join(self.dataFolder, split + '.txt')) as f:
            for line in f:
                line_split = line.split()
                scene_id = line_split[0]
                frame_num = line_split[1]

                if scene_id not in scene_id_to_idx:
                    scenePath = self.dataFolder + '/scenes/' + scene_id
                    scene = ScenenetRgbdScene(options,
                                              scenePath,
                                              scene_id,
                                              load_annotation=self.load_annotation,
                                              load_normals=self.load_normals,
                                              load_scores=self.load_scores,
                                              filter_depth=self.filter_depth,
                                              annotation_dir=annotation_dir,
                                              writer=self.writer)
                    self.scenes.append(scene)

                    scene_id_to_idx[scene_id] = len(self.scenes) - 1

                    scene_id_sc = scene.get_scene_id_sc()
                    # assign each pair of scene_id, scene_plane_id a global plane_id
                    for idx, plane in enumerate(scene.planes):
                        if np.linalg.norm(plane[0:3]) > 1.0e-4:
                            if (scene_id_sc, scene.plane_ids[idx]) not in self.scene_id_sc_plane_to_plane_id:
                                self.scene_id_sc_plane_to_plane_id[(scene_id_sc, scene.plane_ids[idx])] = next_plane_id
                                next_plane_id += 1

                self.sceneImageIndices += [[scene_id_to_idx[scene_id], frame_num]]

        if random:
            t = int(time.time() * 1000000)
            np.random.seed(((t & 0xff000000) >> 24) +
                           ((t & 0x00ff0000) >> 8) +
                           ((t & 0x0000ff00) << 8) +
                           ((t & 0x000000ff) << 24))
        else:
            np.random.seed(0)
            pass
        np.random.shuffle(self.sceneImageIndices)

        # if split == 'test':
        #     with open('/mnt/data/datasets/JW/scenenet_rgbd/test_500.txt', 'w') as tf:
        #         for i in range(500):
        #             tf.write('%s %s\n' % (self.scenes[self.sceneImageIndices[i][0]].scene_id, self.sceneImageIndices[i][1]))

        print('num images', len(self.sceneImageIndices))

        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      config.BACKBONE_SHAPES,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)

    def __len__(self):
        return len(self.sceneImageIndices)

    def get_num_plane_ids(self):
        return len(self.scene_id_sc_plane_to_plane_id)
    
    def __getitem__(self, index_cam_idx):
        raise ValueError('not supported')


def load_image_gt(config, image_id, image, depth, mask, class_ids, parameters,
                  use_mini_mask=True):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    ## Load image and mask
    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MAX_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)

    mask = utils.resize_mask(mask, scale, padding)

    ## Bounding boxes. Note that some boxes might be all zeros
    ## if the corresponding mask got cropped out.
    ## bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    ## Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
        pass

    active_class_ids = np.ones(config.NUM_CLASSES, dtype=np.int32)
    ## Image meta data
    image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

    return image, image_meta, class_ids, bbox, mask, parameters


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    ## RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    ## RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    if gt_boxes.shape[0] > 0:
        ## Handle COCO crowds
        ## A crowd box in COCO is a bounding box around several instances. Exclude
        ## them from training. A crowd box is given a negative class ID.
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

        ## Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = utils.compute_overlaps(anchors, gt_boxes)

        ## Match anchors to GT Boxes
        ## If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
        ## If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
        ## Neutral anchors are those that don't match the conditions above,
        ## and they don't influence the loss function.
        ## However, don't keep any GT box unmatched (rare, but happens). Instead,
        ## match it to the closest anchor (even if its max IoU is < 0.3).
        #
        ## 1. Set negative anchors first. They get overwritten below if a GT box is
        ## matched to them. Skip boxes in crowd areas.
        anchor_iou_argmax = np.argmax(overlaps, axis=1)
        anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
        rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
        ## 2. Set an anchor for each GT box (regardless of IoU value).
        ## TODO: If multiple anchors have the same IoU match all of them
        gt_iou_argmax = np.argmax(overlaps, axis=0)
        rpn_match[gt_iou_argmax] = 1
        ## 3. Set anchors with high overlap as positive.
        rpn_match[anchor_iou_max >= 0.7] = 1

        ## Subsample to balance positive and negative anchors
        ## Don't let positives be more than half the anchors
        ids = np.where(rpn_match == 1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
        if extra > 0:
            ## Reset the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0
        ## Same for negative proposals
        ids = np.where(rpn_match == -1)[0]
        extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                            np.sum(rpn_match == 1))
        if extra > 0:
            ## Rest the extra ones to neutral
            ids = np.random.choice(ids, extra, replace=False)
            rpn_match[ids] = 0

        ## For positive anchors, compute shift and scale needed to transform them
        ## to match the corresponding GT boxes.
        ids = np.where(rpn_match == 1)[0]
        ix = 0  ## index into rpn_bbox
        ## TODO: use box_refinment() rather than duplicating the code here
        for i, a in zip(ids, anchors[ids]):
            ## Closest gt box (it might have IoU < 0.7)
            gt = gt_boxes[anchor_iou_argmax[i]]

            ## Convert coordinates to center plus width/height.
            ## GT Box
            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_center_y = gt[0] + 0.5 * gt_h
            gt_center_x = gt[1] + 0.5 * gt_w
            ## Anchor
            a_h = a[2] - a[0]
            a_w = a[3] - a[1]
            a_center_y = a[0] + 0.5 * a_h
            a_center_x = a[1] + 0.5 * a_w

            ## Compute the bbox refinement that the RPN should predict.
            rpn_bbox[ix] = [
                (gt_center_y - a_center_y) / a_h,
                (gt_center_x - a_center_x) / a_w,
                np.log(gt_h / a_h),
                np.log(gt_w / a_w),
            ]
            ## Normalize
            rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
            ix += 1

    return rpn_match, rpn_bbox
