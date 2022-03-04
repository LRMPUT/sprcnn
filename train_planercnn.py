import cv2
cv2.setNumThreads(0)

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from pytorch_lightning.profiler import PyTorchProfiler

from models.model import *
from datasets.scenenet_rgbd_stereo_dataset import *

from options import parse_args
from config import PlaneConfig


def train(options):
    config = PlaneConfig(options)
    if options.no_normals:
        dataset = ScenenetRgbdDataset(options,
                                           config,
                                           split='train',
                                           random=False,
                                           load_normals=False)
    else:
        dataset = ScenenetRgbdDataset(options, config, split='train', random=False)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    if options.no_normals:
        dataset_test = ScenenetRgbdDataset(options,
                                           config,
                                           split='test',
                                           random=False,
                                           load_normals=False)
    else:
        dataset_test = ScenenetRgbdDataset(options, config, split='test', random=False)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    print('num_plane_ids: %d' % dataset.get_num_plane_ids())
    options.num_plane_ids = dataset.get_num_plane_ids()

    # profiler = PyTorchProfiler(use_cuda=True, profile_memory=True, record_shapes=True)
    # profiler = SimpleProfiler()
    model = MaskRCNN(config, options)
    model.load_weights(['/mnt/data/datasets/JW/scenenet_rgbd/checkpoint/mask_rcnn_coco.pth',
                        '/mnt/data/datasets/JW/scenenet_rgbd/checkpoint/mvdnet_scannet.pth'])
    # model = MaskRCNN.load_from_checkpoint('/mnt/data/datasets/JW/scenenet_rgbd/checkpoint/mask_rcnn_mvdnet.ckpt',
    #                                       config=config,
    #                                       options=options,
    #                                       detect=True,
    #                                       strict=False)

    if options.checkpoint == '':
        trainer = pl.Trainer(gpus=1, max_epochs=options.numEpochs)
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=options.numEpochs,
                             resume_from_checkpoint=options.checkpoint)

    trainer.fit(model, train_loader, test_loader)

    # print(profiler.key_averages().table(sort_by="self_gpu_memory_usage", row_limit=10))


if __name__ == '__main__':
    args = parse_args()

    print('task=%s started' % (args.task))

    train(args)
