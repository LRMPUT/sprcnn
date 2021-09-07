import cv2
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.profiler import AdvancedProfiler, PassThroughProfiler

from models.model import *
from datasets.scenenet_rgbd_stereo_dataset import *

from options import parse_args
from config import PlaneConfig


def eval(options):
    config = PlaneConfig(options)

    if options.no_normals:
        dataset_test = ScenenetRgbdDataset(options,
                                           config,
                                           split='test',
                                           random=False,
                                           load_annotation=True,
                                           load_normals=False,
                                           filter_depth=False,
                                           annotation_dir=options.annotation_dir,
                                           crop_ratio=options.crop_ratio)
    else:
        dataset_test = ScenenetRgbdDataset(options,
                                           config,
                                           split='test',
                                           random=False,
                                           load_annotation=True,
                                           load_normals=True,
                                           filter_depth=True,
                                           annotation_dir=options.annotation_dir,
                                           crop_ratio=options.crop_ratio)

    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

    # profiler = AdvancedProfiler()
    profiler = PassThroughProfiler()
    # model = MaskRCNN(config, options)
    model = MaskRCNN.load_from_checkpoint(options.checkpoint,
                                          config=config,
                                          options=options,
                                          annotations_as_detections=True,
                                          profiler=profiler)
    trainer = pl.Trainer(gpus=1, profiler=profiler)

    trainer.test(model, test_dataloaders=test_loader)


if __name__ == '__main__':
    args = parse_args()

    print('task=%s started' % (args.task))

    eval(args)
