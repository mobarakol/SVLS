import os
import random
import numpy as np
import torch
from torch import nn
from surface_distance import surface_distance

class DataAugmenter(nn.Module):
    """Performs random flip and rotation batch wise, and reverse it if needed.
    Works"""

    def __init__(self, p=0.5):
        super(DataAugmenter, self).__init__()
        self.p = p
        self.transpose = []
        self.flip = []
        self.toggle = False

    def forward(self, x):
        with torch.no_grad():
            if random.random() < self.p:
                self.transpose = random.sample(range(2, x.dim()), 2)
                self.flip = random.randint(2, x.dim() - 1)
                self.toggle = not self.toggle
                new_x = x.transpose(*self.transpose).flip(self.flip)
                return new_x
            else:
                return x

    def reverse(self, x):
        if self.toggle:
            self.toggle = not self.toggle
            return x.flip(self.flip).transpose(*self.transpose)
        else:
            return x


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def surface_dice_3D(im1, im2, tid, spacing_mm=(1,1,1), sd_tolerance = [2]):
    im1 = im1 == tid
    im2 = im2 == tid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # compute surface dice
    surface_dist = surface_distance.compute_surface_distances(
        im1, im2, spacing_mm=spacing_mm)
    sd = []
    for tol_weight in sd_tolerance:
        if len(np.unique(im1)) == 1:
            sd_gt2pred = 0
        else:
            sd_gt2pred, _ = surface_distance.compute_surface_overlap_at_tolerance(surface_dist, tol_weight)
        sd.append(sd_gt2pred)
    return sd


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.device = "cpu"

    def binary_dice(self, inputs, targets, metric_mode=False):
        smooth = 1.
        if metric_mode:
            if targets.sum() == 0:
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
        # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric_brats(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []           
            dice.append(self.binary_dice(inputs[j]==3, target[j]==3, True))
            dice.append(self.binary_dice(torch.logical_or(inputs[j]==1, inputs[j]==3), torch.logical_or(target[j]==1, target[j]==3), True))
            dice.append(self.binary_dice(inputs[j]>0, target[j]>0, True))
            dices.append(dice)
            
        return dices

    def get_surface_dice(self, inputs, target):
        s_dices = []
        for j in range(target.shape[0]):
            dice = []           
            dice.append(surface_dice_3D(inputs[j]==3, target[j]==3, 1))
            dice.append(surface_dice_3D(np.logical_or(inputs[j]==1, inputs[j]==3), np.logical_or(target[j]==1, target[j]==3), 1))
            dice.append(surface_dice_3D(inputs[j]>0, target[j]>0, 1))
            s_dices.append(dice)
            
        return s_dices