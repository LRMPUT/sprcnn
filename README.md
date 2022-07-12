# Stereo Plane R-CNN

![schematic](https://github.com/LRMPUT/sprcnn/raw/master/images/sprcnn.jpg)

## Paper

If you find Stereo Plane R-CNN useful in your academic work please cite the following paper:

    @article{wietrzykowski2022,
        author={Wietrzykowski, Jan and Belter, Dominik},
        journal={IEEE Robotics and Automation Letters}, 
        title={{Stereo Plane R-CNN: Accurate Scene Geometry Reconstruction Using Planar Segments and Camera-Agnostic Representation}}, 
        year={2022},
        volume={7},
        number={2},
        pages={4345-4352},
        doi={10.1109/LRA.2022.3150841}
     }

## Instalation

Clone the repository:
```bash
git clone https://github.com/LRMPUT/sprcnn.git
```
Create a new Conda environment and install required dependencies:
```bash
conda install pytorch torchvision torchaudio torchmetrics cudatoolkit=11.1 numpy scikit-image scipy numpy -c pytorch-lts -c nvidia
```
Install other depencencies using Pip:
```bash
pip install opencv-python disjoint-set pytorch-lightning
```

## Dataset

The `SceneNet Stereo` dataset can be downloaded [here](https://putpoznanpl-my.sharepoint.com/:f:/g/personal/jan_wietrzykowski_put_poznan_pl/ErZm6If9-91JtW7BEK4pXJcBWKLwhoujwisDu_tLDjik2Q?e=vgnlnM).

The `TERRINet` dataset can be downloaded [here](https://putpoznanpl-my.sharepoint.com/:f:/g/personal/jan_wietrzykowski_put_poznan_pl/Eqj0TnSgDrlJuJu0FC-bVGEB2hbpWHC_YA_l_qs9EDgkjw?e=JYRBIT).

## Training

```bash
python train_planercnn.py --dataFolder=/mnt/data/datasets/scenenet_rgbd --anchorType=none_exp_plane_params --normWeight=100.0 --dispWeight=1.0 --LR=0.00001 --numEpochs=10
```

## Inference

Pretrained model can be downloaded [here](https://putpoznanpl-my.sharepoint.com/:f:/g/personal/jan_wietrzykowski_put_poznan_pl/EgQqJVYSoLNJrkw3qWLqZ9IBUf8M0shHE8XNA74cwEs4ww?e=FyjB4g).

```bash
python evaluate.py --anchorType=none_exp_plane_params --dataFolder=/mnt/data/datasets/TERRINet --checkpoint=/mnt/data/datasets/scenenet_rgbd/checkpoint/plane_params.ckpt --no_normals
```

## Exporting detections for PlaneLoc2

Before running the following command, make sure that contents of `annotations_plane_params_det` dir in each scene directory is empty. 

```bash
python evaluate.py --anchorType=none_exp_plane_params --dataFolder=/mnt/data/datasets/TERRINet --checkpoint=/mnt/data/datasets/scenenet_rgbd/checkpoint/plane_params.ckpt --no_normals --export_detections
```
Detections should be exported to `annotations_plane_params_det` dir in each scene directory.

## Acknowledgement

Our implementation is based on [Plane R-CNN](https://github.com/NVlabs/planercnn).
