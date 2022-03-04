"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
cv2.setNumThreads(0)
import torch
import numpy as np
import glob
import os
import re

import utils_cpp_py

from utils import *


class ScenenetRgbdScene():
    """ This class handle one scene of the scannet dataset and provide interface for dataloaders """

    def __init__(self,
                 options,
                 scenePath,
                 scene_id,
                 load_annotation=True,
                 load_normals=True,
                 load_scores=False,
                 filter_depth=True,
                 annotation_dir='annotation',
                 writer=None):

        self.options = options
        self.load_annotation = load_annotation
        self.load_normals = load_normals
        self.load_scores = load_scores
        self.filter_depth = filter_depth
        self.scannetVersion = 2
        self.scene_id = scene_id

        # self.annotation_dir = 'annotation'
        # self.annotation_dir = 'annotation_plane_params'
        self.annotation_dir = annotation_dir

        self.cams = ['left', 'right']

        self.writer = writer

        self.camera = np.zeros(7)

        self.baseline = None
        with open(scenePath + '/' + scene_id + '.txt') as f:
            for line in f:
                line = line.strip()
                tokens = [token for token in line.split(' ') if token.strip() != '']
                if tokens[0] == "fx_depth":
                    self.camera[0] = float(tokens[2])
                if tokens[0] == "fy_depth":
                    self.camera[1] = float(tokens[2])
                if tokens[0] == "mx_depth":
                    self.camera[2] = float(tokens[2])
                if tokens[0] == "my_depth":
                    self.camera[3] = float(tokens[2])
                elif tokens[0] == "colorWidth":
                    self.colorWidth = int(tokens[2])
                elif tokens[0] == "colorHeight":
                    self.colorHeight = int(tokens[2])
                elif tokens[0] == "depthWidth":
                    self.depthWidth = int(tokens[2])
                elif tokens[0] == "depthHeight":
                    self.depthHeight = int(tokens[2])
                elif tokens[0] == "numDepthFrames":
                    self.numImages = int(tokens[2])
                elif tokens[0] == "baseline":
                    self.baseline = float(tokens[2])

        # baseline
        if self.baseline is None:
            self.camera[6] = 0.2
        else:
            self.camera[6] = self.baseline

        self.depthShift = 1000.0
        # self.imagePaths = [scenePath + '/frames/color/' + str(imageIndex) + '.jpg' for imageIndex in range(self.numImages - 1)]
        # self.imagePaths = sorted(glob.glob(scenePath + '/frames/color_left/*.jpg'))
        self.frame_nums = {re.split('[\\/.]', path)[-2] for path in
                           os.listdir(os.path.join(scenePath, 'frames', 'color_left'))}

        # self.imagePaths = [[os.path.join(scenePath, 'frames', 'color_left', frame_num + '.jpg') for frame_num in self.frame_nums],
        #                    [os.path.join(scenePath, 'frames', 'color_right', frame_num + '.jpg') for frame_num in self.frame_nums]]

        self.camera[4] = self.depthWidth
        self.camera[5] = self.depthHeight

        if self.load_annotation:
            self.planes = np.load(scenePath + '/' + self.annotation_dir + '/planes.npy')
            self.plane_ids = np.arange(self.planes.shape[0], dtype=np.int)

            merge_file = os.path.join(scenePath, self.annotation_dir, 'merge.txt')
            if os.path.exists(merge_file):
                self.plane_ids = np.loadtxt(merge_file, dtype=np.int)
        else:
            self.planes = np.zeros((0, 4))
            self.plane_ids = -1 * np.ones(0, dtype=np.int)

        self.scenePath = scenePath
        return

    def get_scene_id_sc(self):
        return '_'.join(self.scene_id.split('_')[:-1])

    def get_unique_plane_ids(self):
        return np.unique(self.plane_ids)

    def transformPlanes(self, transformation, planes):
        if planes.shape[1] == 3:
            planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)

            centers = planes
            centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1)
            newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
            newCenters = newCenters[:, :3] / newCenters[:, 3:4]

            refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
            refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
            newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
            newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

            planeNormals = newRefPoints - newCenters
            planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
            planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
            newPlanes = planeNormals * planeOffsets
            return newPlanes
        else:
            planes_camera = (np.linalg.inv(transformation.transpose()) @ planes.transpose()).transpose()
            planes_camera = planes_camera[:, 0:3] * (-planes_camera[:, 3:4])
            return planes_camera

    def makeBoxySegmentation(self, segmentation):
        boxy_segm, new_id_to_old_id = utils_cpp_py.comp_components(segmentation, 200, 0.5)

        segments, counts = np.unique(boxy_segm, return_counts=True)
        segmentList = zip(segments.tolist(), counts.tolist())
        segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]
        segmentList = sorted(segmentList, key=lambda x: -x[1])

        newPlanes = []
        newPlaneInfo = []
        newSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)

        newIndex = 0
        for new_id, count in segmentList:
            if new_id_to_old_id[new_id] >= 0:
                old_id = new_id_to_old_id[new_id]
                if count < self.options.planeAreaThreshold:
                    continue
                if old_id >= len(self.planes):
                    continue
                if np.linalg.norm(self.planes[old_id]) < 1e-4:
                    continue
                newPlanes.append(self.planes[old_id])
                newSegmentation[boxy_segm == new_id] = newIndex
                newPlaneInfo.append(old_id)
                newIndex += 1
                continue

        return newSegmentation, newPlanes, newPlaneInfo

    def __len__(self):
        return len(self.frame_nums)

    def __getitem__(self, frame_num_cam_idx):
        frame_num = frame_num_cam_idx[0]
        cam_idx = frame_num_cam_idx[1]

        imagePath = os.path.join(self.scenePath, 'frames', 'color_' + self.cams[cam_idx], frame_num + '.jpg')
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentationPath = imagePath.replace('frames/color_' + self.cams[cam_idx] + '/',
                                             self.annotation_dir + '/segmentation_' + self.cams[cam_idx] + '/').replace('.jpg',
                                                                                                            '.png')
        depthPath = imagePath.replace('color', 'depth').replace('.jpg', '.png')
        normXPath = imagePath.replace('color', 'norm').replace('.jpg', '_x.png')
        normYPath = imagePath.replace('color', 'norm').replace('.jpg', '_y.png')
        normZPath = imagePath.replace('color', 'norm').replace('.jpg', '_z.png')
        posePath = imagePath.replace('color', 'pose').replace('.jpg', '.txt')
        scoresPath = imagePath.replace('frames/color_' + self.cams[cam_idx] + '/',
                                       self.annotation_dir + '/scores_' + self.cams[cam_idx] + '/').replace('.jpg', '.npz')

        depth = cv2.imread(depthPath, -1).astype(np.float32) / self.depthShift
        if self.load_normals:
            norm_x = cv2.imread(normXPath, -1).astype(np.float32) / 10000.0 - 1.0
            norm_y = cv2.imread(normYPath, -1).astype(np.float32) / 10000.0 - 1.0
            norm_z = cv2.imread(normZPath, -1).astype(np.float32) / 10000.0 - 1.0
            normal = np.stack([norm_x, norm_y, norm_z], axis=-1)
            # assign 0 to invalid normals
            normal = np.where(np.abs(np.linalg.norm(normal, axis=-1, keepdims=True) - 1.0) < 1.0e-3, normal, 0.0)

            ranges = get_ranges(self.camera)

            # dot product > 0
            wrong_sign = np.sum(ranges * normal, axis=2) > 0
            normal[wrong_sign] = -normal[wrong_sign]
        else:
            normal = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)

        extrinsics_inv = []
        with open(posePath, 'r') as f:
            for line in f:
                extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                continue
            pass
        extrinsics_inv = np.array(extrinsics_inv).reshape((4, 4))
        extrinsics = np.linalg.inv(extrinsics_inv)

        if self.load_annotation:
            segmentation = cv2.imread(segmentationPath, -1).astype(np.int32)

            segmentation = (segmentation[:, :, 2] * 256 * 256 +
                            segmentation[:, :, 1] * 256 +
                            segmentation[:, :, 0]) // 100 - 1

            newSegmentation, newPlanes, newPlaneInfo = self.makeBoxySegmentation(segmentation)

            segmentation = newSegmentation
            planes_global = np.array(newPlanes)
            plane_info = np.array(newPlaneInfo, dtype=np.int32)
            # convert to merged plane ids
            plane_info = self.plane_ids[plane_info]

            image = cv2.resize(image, (depth.shape[1], depth.shape[0]))

            planes = np.zeros((0, 3), dtype=np.float32)
            if len(planes_global) > 0:
                planes = self.transformPlanes(extrinsics, planes_global)
                segmentation, plane_depths = cleanSegmentation(image, planes, segmentation, depth, self.camera,
                                                               planeAreaThreshold=self.options.planeAreaThreshold,
                                                               planeWidthThreshold=self.options.planeWidthThreshold,
                                                               depthDiffThreshold=0.1 if self.filter_depth else 1e6,
                                                               return_plane_depths=True)

                masks = (np.expand_dims(segmentation, -1) == np.arange(len(planes))).astype(np.float32)
                plane_depth = (plane_depths.transpose((1, 2, 0)) * masks).sum(2)
                plane_mask = masks.max(2)
                plane_mask *= (depth > 1e-4).astype(np.float32)
                plane_area = plane_mask.sum()
                depth_error = (np.abs(plane_depth - depth) * plane_mask).sum() / max(plane_area, 1)

                if self.filter_depth and depth_error > 0.1:
                    # if self.writer is not None:
                    #     self.writer.add_image('scene/image', image, dataformats='HWC')
                    #     # up to 15 m
                    #     self.writer.add_image('scene/depth', depth / 15.0, dataformats='HW')
                    #     self.writer.add_image('scene/plane_depth', plane_depth / 15.0, dataformats='HW')
                    #     self.writer.add_image('scene/plane_mask', plane_mask, dataformats='HW')
                    #     self.writer.add_image('scene/depth_error', (np.abs(plane_depth - depth) * plane_mask) / 3.0,
                    #                           dataformats='HW')
                    #     self.writer.add_text('scene/name', self.scene_id + ' ' + frame_num + ' ' + str(cam_idx))
                    #     print(self.scene_id + ' ' + frame_num + ' ' + str(cam_idx))
                    #     # print(planes)
                    #     # print(planes_global)
                    #     # self.writer.flush()
                    #     pass

                    print('depth error', depth_error)
                    planes = np.zeros((0, 3), dtype=np.float32)
                    plane_info = np.zeros(0, dtype=np.float32)

            # if len(planes) == 0 or segmentation.max() < 0:
            #     raise Exception('No planes on the view')
        else:
            planes = np.zeros((0, 3), dtype=np.float32)
            plane_info = np.zeros(0, dtype=np.float32)
            segmentation = -1 * np.ones((depth.shape[0], depth.shape[1]), dtype=np.int32)

        semantics = np.zeros_like(depth)

        info = [image, planes, segmentation, depth, normal, self.camera, extrinsics, semantics, plane_info]

        if self.load_scores:
            anchor_data = np.load(scoresPath)
            scores_a = anchor_data['scores']
            planes_a = anchor_data['planes']
            masks_a = anchor_data['masks']

            info.extend([scores_a, planes_a, masks_a])

        return info
