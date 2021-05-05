import argparse
import os
import numpy as np
import pathlib
import torch
from model import UNet3D
from datasets import get_datasets_brats
from utils import seed_everything, EDiceLoss
        
def step_valid(data_loader, model, metric):
    model.eval()
    metrics = []
    for i, batch in enumerate(data_loader):
        targets = batch["label"].squeeze(1).cuda(non_blocking=True)
        inputs = batch["image"].float().cuda()
        segs = model(inputs)
        segs = segs.data.max(1)[1].squeeze_(1)
        metric_ = metric(segs.detach().cpu(), targets.detach().cpu())
        metrics.extend(metric_)
    return metrics
    
def main():
    parser = argparse.ArgumentParser(description='SVLS Brats Training')
    parser.add_argument('--batch_size', default=2, type=int,help='mini-batch size')
    parser.add_argument('--num_classes', default=4, type=int, help="num of classes")
    parser.add_argument('--in_channels', default=4, type=int, help="num of input channels")
    parser.add_argument('--train_option', default='SVLS', help="options:[SVLS, LS, OH]")
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--data_root', default='MICCAI_BraTS_2019_Data_Training/HGG_LGG', help='data directory')
    parser.add_argument('--ckpt_dir', default='ckpt_brats19', help='ckpt directory')
    args = parser.parse_args() 

    _, val_dataset = get_datasets_brats(data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, num_workers=2)

    print('valid sample:',len(val_dataset), 'valid minibatch:',len(val_loader))

    model = UNet3D(inplanes=args.in_channels, num_classes=args.num_classes).cuda()

    model = torch.nn.DataParallel(model)
    criterion_dice = EDiceLoss().cuda()
    
    print('train_option',args.train_option)
    if args.train_option == 'SVLS':
        best_ckpt_name = 'best_svls.pth.tar'
    elif args.train_option == 'LS':
        best_ckpt_name = 'best_ls{}'.format(args.ls_smoothing)
    elif args.train_option == 'OH':
        best_ckpt_name = 'best_oh.pth.tar'
    else:
        raise ValueError(args.train_option)
    
    print('ckpt name:', best_ckpt_name)
    best_ckpt_dir = os.path.join(args.ckpt_dir, best_ckpt_name)
    model.load_state_dict(torch.load(best_ckpt_dir))
    metric = criterion_dice.metric_brats
    with torch.no_grad():
        dice_metrics = step_valid(val_loader, model, metric)
        dice_metrics = list(zip(*dice_metrics))
        dice_metrics = [torch.tensor(dice, device="cpu").numpy() for dice in dice_metrics]
        avg_dices = np.mean(dice_metrics,1)

    print('dice[ET:%.4f, TC:%.4f, WT:%.4f]' %(avg_dices[0], avg_dices[1], avg_dices[2]))

if __name__ == '__main__':
    seed_everything()
    main()
