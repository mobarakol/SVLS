import torch
import torch.nn.functional as F
import math


def get_svls_filter_3d(kernel_size=3, sigma=1, channels=4):
    # Create a x, y, z coordinate grid of shape (kernel_size, kernel_size, kernel_size, 3)
    x_coord = torch.arange(kernel_size)
    x_grid_2d = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    x_grid = x_coord.repeat(kernel_size*kernel_size).view(kernel_size, kernel_size, kernel_size)
    y_grid_2d = x_grid_2d.t()
    y_grid  = y_grid_2d.repeat(kernel_size,1).view(kernel_size, kernel_size, kernel_size)
    z_grid = y_grid_2d.repeat(1,kernel_size).view(kernel_size, kernel_size, kernel_size)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    # Calculate the 3-dimensional gaussian kernel
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp(
                            -torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance + 1e-16))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    neighbors_sum = 1 - gaussian_kernel[1,1,1]
    gaussian_kernel[1,1,1] = neighbors_sum
    svls_kernel_3d = gaussian_kernel / neighbors_sum

    # Reshape to 3d depthwise convolutional weight
    svls_kernel_3d = svls_kernel_3d.view(1, 1, kernel_size, kernel_size, kernel_size)
    svls_kernel_3d = svls_kernel_3d.repeat(channels, 1, 1, 1, 1)
    svls_filter_3d = torch.nn.Conv3d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=0)
    svls_filter_3d.weight.data = svls_kernel_3d
    svls_filter_3d.weight.requires_grad = False 
    return svls_filter_3d, svls_kernel_3d[0]

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


class CELossWithLS(torch.nn.Module):
    def __init__(self, classes=None, smoothing=0.0):
        super(CELossWithLS, self).__init__()
        self.complement = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = torch.tensor(classes)
        self.cls_idx = torch.arange(self.cls).reshape(1, self.cls).cuda()

    def forward(self, inputs, labels):
        with torch.no_grad():
            oh_labels = (labels[...,None] == self.cls_idx).permute(0,4,1,2,3)
            smoothen_label = oh_labels * self.complement + self.smoothing / self.cls
        return (- smoothen_label * F.log_softmax(inputs, dim=1)).sum(dim=1).mean()
