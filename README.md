# SVLS

This repository contains the code implementation of the IPMI 2021 paper [Spatially Varying Label Smoothing: Capturing Uncertainty from Expert Annotations](https://arxiv.org/pdf/2104.05788.pdf)<br>


## A simple SVLS implementation
3D as paper:

```python

class CELossWithSVLS(torch.nn.Module):
    def __init__(self, classes=None, sigma=1):
        super(CELossWithSVLS, self).__init__()
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()
        self.svls_layer, self.svls_kernel = get_svls_filter_3d(sigma=sigma, channels=classes)
        self.svls_kernel = self.svls_kernel.cuda()

    def forward(self, inputs, labels):
        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            b, c, d, h, w = oh_labels.shape
            x = oh_labels.view(b, c, d, h, w).repeat(1, 1, 1, 1, 1).float()
            x = F.pad(x, (1,1,1,1,1,1), mode='replicate')
            svls_labels = self.svls_layer(x)/self.svls_kernel.sum()
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()
```
2D version for 2D image segmentation: you can also find an interactive Colab notebook on toy example for 2D SVLS implementation. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/mobarakol/SVLS/blob/main/SVLS_2D.ipynb)

```
class CELossWithSVLS_2D(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, ksize=3):
        super(CELossWithSVLS_2D, self).__init__()
        self.cls = torch.tensor(classes)
        self.svls_layer = get_svls_filter_2d(ksize=3, sigma=sigma, channels=self.cls)

    def forward(self, inputs, labels):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()
        svls_labels = self.svls_layer(oh_labels)
        return (- svls_labels * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()
```

The implementation of the surface dice is adopted from this [repository](https://github.com/deepmind/surface-distance) and rename the folder to "surface_distance" to avoid library importing issue in python. <br>

Train command for SVLS on BraTS 2019 <br>

```
CUDA_VISIBLE_DEVICES=0,1 python main.py --batch_size 2 --data_root /vol/biomedic3/mi615/datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG_LGG/ --train_option SVLS
```

Validation command for SVLS on BraTS 2019 <br>

```
CUDA_VISIBLE_DEVICES=0,1 python deploy.py --batch_size 2 --data_root /vol/biomedic3/mi615/datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG_LGG/ --train_option SVLS
```

To run a demo on all training options and evaluation metrics with reliability diagram with BraTS 2019 <br>

```
CUDA_VISIBLE_DEVICES=0,1 python demo.py --batch_size 2 --data_root /vol/biomedic3/mi615/datasets/BraTS/MICCAI_BraTS_2019_Data_Training/HGG_LGG/ --train_option SVLS
```

The model architecture is adopted from this [repository](https://github.com/wolny/pytorch-3dunet)


## Citation
If you use this code for your research, please cite our paper.

```
@article{islam2021spatially,
  title={Spatially Varying Label Smoothing: Capturing Uncertainty from Expert Annotations},
  author={Islam, Mobarakol and Glocker, Ben},
  journal={arXiv preprint arXiv:2104.05788},
  year={2021}
}
```
