# SVLS

This repository contains the code implementation of the IPMI 2021 paper [Spatially Varying Label Smoothing: Capturing Uncertainty from Expert Annotations](https://arxiv.org/pdf/2104.05788.pdf)<br>

The implementation of the surface dice is adopted from this [repository](https://github.com/deepmind/surface-distance) and rename the folder to "surface_distance" to avoid library importing issue in python. <br>

Train command for SVLS on BraTS 2019 <br>

```
CUDA_VISIBLE_DEVICES=0,1 python main.py --batch_size 2 --data_root /vol/biomedic3/mi615/datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG_LGG/ --train_option SVLS
```

Validation command for SVLS on BraTS 2019 <br>

```
CUDA_VISIBLE_DEVICES=0,1 python main.py --batch_size 2 --data_root /vol/biomedic3/mi615/datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG_LGG/ --train_option SVLS
```

To run a demo on all training options with BraTS 2019 <br>

```
CUDA_VISIBLE_DEVICES=0,1 python demo.py --batch_size 2 --data_root /vol/biomedic3/mi615/datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG_LGG/ --train_option SVLS
```

The model architecture adopted from this [repository](https://github.com/wolny/pytorch-3dunet)