"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
cv2.setNumThreads(0)
import torch
import torchvision.transforms.functional as tvf
import torchvision as tv
from torch.utils.data import Dataset

import numpy as np
import time
import utils as utils
import os

from datasets.scenenet_rgbd_scene import ScenenetRgbdScene
from datasets.scenenet_rgbd_dataset import *


class ScenenetRgbdDataset(ScenenetRgbdDatasetSingle):
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
                 crop_ratio=1.0,
                 writer=None):
        super().__init__(options,
                         config,
                         split,
                         random,
                         load_annotation=load_annotation,
                         load_normals=load_normals,
                         load_scores=load_scores,
                         filter_depth=filter_depth,
                         annotation_dir=annotation_dir,
                         writer=writer)

        self.load_annotation = load_annotation
        self.load_normals = load_normals
        self.load_scores = load_scores
        self.filter_depth = filter_depth

        self.crop_ratio = crop_ratio

        self.writer = writer

        self.camera_names = ['left', 'right']

    def jitter_color(self, image, bf, cf, hf, sf):
        image = tvf.adjust_brightness(image, bf)
        image = tvf.adjust_contrast(image, cf)
        # image = tvf.adjust_gamma(image, gf)
        image = tvf.adjust_hue(image, hf)
        image = tvf.adjust_saturation(image, sf)

        return image

    def rand_float(self, low, high):
        return float(torch.rand(1) * (high - low) + low)

    def augment(self, data_pair, augment_image=True, crop_factor=None):
        # image_metas_left = torch.from_numpy(data_pair['left']['image_metas']).unsqueeze(0)
        # image_metas_right = torch.from_numpy(data_pair['right']['image_metas']).unsqueeze(0)
        image_left = utils.to_torch_image_s(data_pair['left']['image']) / 255.0
        image_right = utils.to_torch_image_s(data_pair['right']['image']) / 255.0
        depth_left = utils.to_torch_image_s(data_pair['left']['depth'])
        depth_right = utils.to_torch_image_s(data_pair['right']['depth'])
        segmentation_left = utils.to_torch_image_s(data_pair['left']['segmentation'])
        segmentation_right = utils.to_torch_image_s(data_pair['right']['segmentation'])
        masks_left = utils.to_torch_image_s(data_pair['left']['gt_masks'])
        masks_right = utils.to_torch_image_s(data_pair['right']['gt_masks'])

        im_h = image_left.shape[2]
        im_w = image_left.shape[3]

        brightness_range = 0.1
        contrast_range = 0.1
        # gamma_range = 0.1
        hue_range = 0.1
        saturation_range = 0.1
        sharpness_range = 0.1
        gaussian_range = 0.05
        crop_range = 20

        diff_range = 0.05

        data_pair_aug = data_pair.copy()

        if augment_image:
            # color
            bf = self.rand_float(1 - brightness_range, 1 + brightness_range)
            cf = self.rand_float(1 - contrast_range, 1 + contrast_range)
            # gf = self.rand_float(1 - gamma_range, 1 + gamma_range)
            hf = self.rand_float(-hue_range, hue_range)
            sf = self.rand_float(1 - saturation_range, 1 + saturation_range)
            image_left = self.jitter_color(image_left, bf, cf, hf, sf)
            # different for right image
            # bf = self.rand_float(1 - brightness_range, 1 + brightness_range)
            # cf = self.rand_float(1 - contrast_range, 1 + contrast_range)
            # gf = self.rand_float(1 - gamma_range, 1 + gamma_range)
            # hf = self.rand_float(-hue_range, hue_range)
            # sf = self.rand_float(1 - saturation_range, 1 + saturation_range)
            # slightly different for right image
            bf += self.rand_float(-diff_range * brightness_range, diff_range * brightness_range)
            cf += self.rand_float(-diff_range * contrast_range, diff_range * contrast_range)
            # gf = self.rand_float(1 - gamma_range, 1 + gamma_range)
            hf += self.rand_float(-diff_range * hue_range, diff_range * hue_range)
            sf += self.rand_float(-diff_range * saturation_range, diff_range * saturation_range)
            image_right = self.jitter_color(image_right, bf, cf, hf, sf)

            # sharpness
            shf = self.rand_float(1 - sharpness_range, 1 + sharpness_range)
            image_left = tvf.adjust_sharpness(image_left, shf)
            shf += self.rand_float(diff_range * sharpness_range, diff_range * sharpness_range)
            image_right = tvf.adjust_sharpness(image_right, shf)

            # noise
            gaussf = self.rand_float(0, gaussian_range)
            noise = torch.randn((1, 1, im_h, im_w)).type_as(image_left) * gaussf
            image_left = (image_left + noise).clamp(min=0.0, max=1.0)
            gaussf += self.rand_float(-diff_range * gaussian_range, diff_range * gaussian_range)
            noise = torch.randn((1, 1, im_h, im_w)).type_as(image_right) * gaussf
            image_right = (image_right + noise).clamp(min=0.0, max=1.0)

        # crop and resize
        if crop_factor is None:
            y1 = int(self.rand_float(0, crop_range))
            y2 = int(self.rand_float(im_h - crop_range, im_h))
            x1 = int(self.rand_float(0, crop_range))
            x2 = int(self.rand_float(im_w - crop_range, im_w))
        else:
            # crop symmetrically
            y1 = int((im_h - im_h * crop_factor) / 2)
            y2 = int(im_h - (im_h - im_h * crop_factor) / 2)
            x1 = int((im_w - im_w * crop_factor) / 2)
            x2 = int(im_w - (im_w - im_w * crop_factor) / 2)
        image_left = tvf.resized_crop(image_left,
                                      y1, x1,
                                      y2 - y1, x2 - x1,
                                      (im_h, im_w))
        image_right = tvf.resized_crop(image_right,
                                       y1, x1,
                                       y2 - y1, x2 - x1,
                                       (im_h, im_w))
        depth_left = tvf.resized_crop(depth_left,
                                      y1, x1,
                                      y2 - y1, x2 - x1,
                                      (im_h, im_w))
        depth_right = tvf.resized_crop(depth_right,
                                       y1, x1,
                                       y2 - y1, x2 - x1,
                                       (im_h, im_w))
        segmentation_left = tvf.resized_crop(segmentation_left,
                                             y1, x1,
                                             y2 - y1, x2 - x1,
                                             (im_h, im_w),
                                             interpolation=tvf.InterpolationMode.NEAREST)
        segmentation_right = tvf.resized_crop(segmentation_right,
                                              y1, x1,
                                              y2 - y1, x2 - x1,
                                              (im_h, im_w),
                                              interpolation=tvf.InterpolationMode.NEAREST)
        # if number of planes > 0
        if masks_left.shape[1] > 0:
            masks_left = tvf.resized_crop(masks_left,
                                          y1, x1,
                                          y2 - y1, x2 - x1,
                                          (im_h, im_w),
                                          interpolation=tvf.InterpolationMode.NEAREST)
        if masks_right.shape[1] > 0:
            masks_right = tvf.resized_crop(masks_right,
                                           y1, x1,
                                           y2 - y1, x2 - x1,
                                           (im_h, im_w),
                                           interpolation=tvf.InterpolationMode.NEAREST)

        data_pair_aug['left']['image'] = utils.to_numpy_image_s(image_left * 255.0)
        data_pair_aug['right']['image'] = utils.to_numpy_image_s(image_right * 255.0)
        data_pair_aug['left']['depth'] = utils.to_numpy_image_s(depth_left)
        data_pair_aug['right']['depth'] = utils.to_numpy_image_s(depth_right)
        data_pair_aug['left']['segmentation'] = utils.to_numpy_image_s(segmentation_left)
        data_pair_aug['right']['segmentation'] = utils.to_numpy_image_s(segmentation_right)
        data_pair_aug['left']['gt_masks'] = utils.to_numpy_image_s(masks_left)
        data_pair_aug['right']['gt_masks'] = utils.to_numpy_image_s(masks_right)

        # rpn_bbox are relative to box dimension, so it's not necessary to adjust them

        # fx
        data_pair_aug['camera'][0] *= im_w / (x2 - x1)
        # fy
        data_pair_aug['camera'][1] *= im_h / (y2 - y1)
        # cx
        data_pair_aug['camera'][2] -= x1
        data_pair_aug['camera'][2] *= im_w / (x2 - x1)
        # cy
        data_pair_aug['camera'][3] -= y1
        data_pair_aug['camera'][3] *= im_h / (y2 - y1)

        # remove targets with area 0
        for cam in self.camera_names:
            # if number of planes > 0
            # if data_pair_aug[cam]['gt_masks'].shape[2] > 0:
            valid_idxs = data_pair_aug[cam]['gt_masks'].sum(axis=(0, 1)) > 0
            data_pair_aug[cam]['gt_masks'] = data_pair_aug[cam]['gt_masks'][:, :, valid_idxs]
            data_pair_aug[cam]['gt_class_ids'] = data_pair_aug[cam]['gt_class_ids'][valid_idxs]
            data_pair_aug[cam]['gt_parameters'] = data_pair_aug[cam]['gt_parameters'][valid_idxs, :]

        return data_pair_aug

    def __getitem__(self, index):
        while True:
            if self.random:
                index = np.random.randint(len(self.sceneImageIndices))
            else:
                index = (index + 1) % len(self.sceneImageIndices)
                pass

            sceneIndex, frame_num = self.sceneImageIndices[index]
            scene = self.scenes[sceneIndex]

            try:
                # left image
                info_1 = scene[frame_num, 0]
            except Exception as e:
                print('Exception for first %s %s: ' % (scene.scene_id, frame_num), e)
                continue
            except:
                print('Unknown exception for first %s %s' % (scene.scene_id, frame_num))

            # info_1 = [image_1, planes_1, plane_info_1, segmentation_1, depth_1, camera_1, extrinsics_1, semantics_1]

            try:
                # right image
                info_2 = scene[frame_num, 1]
            except Exception as e:
                print('Exception for second %s %s: ' % (scene.scene_id, frame_num), e)
                continue
            except:
                print('Unknown exception for second %s %s' % (scene.scene_id, frame_num))
                continue

            # info_2 = [image_2, planes_2, plane_info_2, segmentation_2, depth_2, camera_2, extrinsics_2, semantics_2]
            break

        data_pair = {}
        # extrinsics_pair = []
        for idx, info in enumerate([info_1, info_2]):

            if self.load_scores:
                image, planes, segmentation, depth, normal, camera, extrinsics, semantics, scores_a, planes_a, masks_a = info
            else:
                image, planes, segmentation, depth, normal, camera, extrinsics, semantics = info

            image = cv2.resize(image, (depth.shape[1], depth.shape[0]))

            instance_masks = []
            class_ids = []
            parameters = []

            for planeIndex, plane in enumerate(planes):
                m = segmentation == planeIndex
                if m.sum() < 1:
                    continue
                instance_masks.append(m)
                if self.config.ANCHOR_TYPE == 'none' or 'none_exp_' in self.config.ANCHOR_TYPE:
                    class_ids.append(1)
                    parameters.append(np.concatenate([plane, np.zeros(1)], axis=0))

            if len(instance_masks) > 0:
                gt_parameters = np.array(parameters)
                gt_masks = np.stack(instance_masks, axis=2)
                gt_class_ids = np.array(class_ids, dtype=np.int32)
            else:
                gt_parameters = np.zeros((0, self.config.NUM_PARAMETERS + 1))
                gt_masks = np.zeros((segmentation.shape[0], segmentation.shape[1], 0), dtype=np.bool)
                gt_class_ids = np.array([], dtype=np.int32)

            # image = utils.mold_image(image.astype(np.float32), self.config)
            image = image.astype(np.float32)

            depth = np.concatenate([depth[:, :, None], normal], axis=-1)

            gt_masks = gt_masks.astype(np.uint8)
            gt_parameters = gt_parameters.astype(np.float32)
            extrinsics = extrinsics.astype(np.float32)
            planes = planes.astype(np.float32)
            segmentation = segmentation[:, :, None]

            data_pair[self.camera_names[idx]] = {'image': image,
                                                 'gt_class_ids': gt_class_ids,
                                                 'gt_masks': gt_masks,
                                                 'gt_parameters': gt_parameters,
                                                 'depth': depth,
                                                 'extrinsics': extrinsics,
                                                 'planes': planes,
                                                 'segmentation': segmentation,
                                                 }

            if self.load_scores:
                data_pair[self.camera_names[idx]]['scores_a'] = scores_a
                data_pair[self.camera_names[idx]]['planes_a'] = planes_a

            # extrinsics_pair.append(extrinsics)
            continue

        data_pair['camera'] = camera.astype(np.float32)
        data_pair['scene_id'] = scene.scene_id
        data_pair['frame_num'] = frame_num

        if self.split == 'train':
            data_pair = self.augment(data_pair)
        elif self.split == 'test' and self.crop_ratio != 1.0:
            data_pair = self.augment(data_pair, augment_image=False, crop_factor=self.crop_ratio)

        for cam in self.camera_names:
            [image,
             image_metas,
             gt_class_ids,
             gt_boxes,
             gt_masks,
             gt_parameters] = load_image_gt(self.config, index,
                                            data_pair[cam]['image'],
                                            data_pair[cam]['depth'],
                                            data_pair[cam]['gt_masks'],
                                            data_pair[cam]['gt_class_ids'],
                                            data_pair[cam]['gt_parameters'])

            ## RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                    gt_class_ids, gt_boxes, self.config)

            ## If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                        np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]
                gt_parameters = gt_parameters[ids]
                pass

            ## Add to batch
            rpn_match = rpn_match[:, np.newaxis]

            ## Convert
            # squeeze, because dataloader adds batch dimension
            image = utils.to_torch_image_s(image).squeeze(0)
            depth = utils.to_torch_image_s(utils.pad_image(data_pair[cam]['depth'],
                                                           image_metas,
                                                           self.config,
                                                           0)).squeeze(0)
            segmentation = utils.to_torch_image_s(utils.pad_image(data_pair[cam]['segmentation'],
                                                                  image_metas,
                                                                  self.config,
                                                                  -1)).squeeze(0)

            image_metas = torch.from_numpy(image_metas)
            rpn_match = torch.from_numpy(rpn_match)
            rpn_bbox = torch.from_numpy(rpn_bbox).float()
            gt_class_ids = utils.pad_zeros_torch(torch.from_numpy(gt_class_ids), self.config.MAX_GT_INSTANCES)
            gt_boxes = utils.pad_zeros_torch(torch.from_numpy(gt_boxes).float(), self.config.MAX_GT_INSTANCES)
            gt_masks = utils.pad_zeros_torch(torch.from_numpy(gt_masks.astype(np.float32).transpose(2, 0, 1)),
                                             self.config.MAX_GT_INSTANCES)
            planes = utils.pad_zeros_torch(torch.from_numpy(data_pair[cam]['planes']), self.config.MAX_GT_INSTANCES)
            plane_indices = utils.pad_zeros_torch(torch.from_numpy(gt_parameters[:, -1]).long(),
                                                  self.config.MAX_GT_INSTANCES)
            gt_parameters = utils.pad_zeros_torch(torch.from_numpy(gt_parameters[:, :-1]).float(),
                                                  self.config.MAX_GT_INSTANCES)

            extrinsics = torch.from_numpy(data_pair[cam]['extrinsics'])

            data_pair[cam]['image'] = image
            data_pair[cam]['image_metas'] = image_metas
            data_pair[cam]['rpn_match'] = rpn_match
            data_pair[cam]['rpn_bbox'] = rpn_bbox
            data_pair[cam]['gt_class_ids'] = gt_class_ids
            data_pair[cam]['gt_boxes'] = gt_boxes
            data_pair[cam]['gt_masks'] = gt_masks
            data_pair[cam]['gt_parameters'] = gt_parameters
            data_pair[cam]['depth'] = depth
            data_pair[cam]['extrinsics'] = extrinsics
            data_pair[cam]['planes'] = planes
            data_pair[cam]['segmentation'] = segmentation
            data_pair[cam]['plane_indices'] = plane_indices

        data_pair['camera'] = torch.from_numpy(data_pair['camera'])

        return data_pair
