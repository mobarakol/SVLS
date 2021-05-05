import argparse
import os
import numpy as np
import pathlib
import torch
from torch.nn import functional as F
from model import UNet3D
from datasets import get_datasets_brats
from utils import seed_everything, EDiceLoss
from calibration_metrics import ece_eval, tace_eval, reliability_diagram
import warnings
warnings.filterwarnings("ignore")
        
def step_valid(data_loader, model, metric):
    ece_all, acc_all, conf_all, tace_all = [], [], [], []
    losses, metrics, metrics_sd = [], [], []
    model.eval()
    for i, batch in enumerate(data_loader):
        targets = batch["label"].squeeze(1).cuda(non_blocking=True)
        inputs = batch["image"].float().cuda()
        segs = model(inputs)
        outputs = F.softmax(segs, dim=1).detach().cpu().numpy()
        if len(targets.shape) < 4:#if batch size=1
            targets = targets.unsqueeze(0)
        labels = targets.detach().cpu().numpy()
        
        ece, acc, conf, _ = ece_eval(outputs,labels)
        tace, _, _, _ = tace_eval(outputs,labels)
        ece_all.append(ece)
        acc_all.append(acc)
        conf_all.append(conf)
        tace_all.append(tace)
        segs = segs.data.max(1)[1].squeeze_(1)
        metric_ = metric.metric_brats(segs, targets)
        metrics_sd.extend(metric.get_surface_dice(segs.detach().cpu().numpy(), targets.detach().cpu().numpy()))
        metrics.extend(metric_)

    ece_avg = np.stack(ece_all).mean(0)
    acc_avg = np.stack(acc_all).mean(0)
    conf_avg = np.stack(conf_all).mean(0)
    tace_avg = np.stack(tace_all).mean(0)
    return metrics, metrics_sd, ece_avg, acc_avg, conf_avg, tace_avg
    
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
    
    legends = ['OH', 'LS(0.1)', 'LS(0.2)', 'LS(0.3)', 'SVLS']
    model_list = ['best_oh.pth.tar', 'best_ls0.1.pth.tar', 'best_ls0.2.pth.tar', 'best_ls0.3.pth.tar', 'best_svls.pth.tar']
    for model_name, legend in zip(model_list, legends):
        model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, model_name)))
        model.eval()
        with torch.no_grad():
            dice_metrics, metrics_sd, ece_avg, acc_avg, conf_avg, tace_avg = step_valid(val_loader, model, criterion_dice)
        if legend != 'LS(0.3)':
            reliability_diagram(conf_avg, acc_avg, legend=legend)
        dice_metrics = list(zip(*dice_metrics))
        dice_metrics = [torch.tensor(dice, device="cpu").numpy() for dice in dice_metrics]
        avg_dices = np.mean(dice_metrics,1)
        avg_std = np.std(dice_metrics,1)

        metrics_sd = list(zip(*metrics_sd))
        metrics_sd = [torch.tensor(dice, device="cpu").numpy() for dice in metrics_sd]
        avg_sd = np.mean(metrics_sd,1)
        avg_std_sd = np.std(metrics_sd,1)

        print('model:%s , dice[ET:%.3f ± %.3f, TC:%.3f ± %.3f, WT:%.3f ± %.3f], ECE:%.4f, TACE:%.4f'%(
            model_name, avg_dices[0],avg_std[0], avg_dices[1],avg_std[1], avg_dices[2],avg_std[2], ece_avg, tace_avg))
        
        print('model:%s , Surface dice[ET:%.3f ± %.3f, TC:%.3f ± %.3f, WT:%.3f ± %.3f]'%(
            model_name, avg_sd[0],avg_std_sd[0], avg_sd[1],avg_std_sd[1], avg_sd[2],avg_std_sd[2]))

if __name__ == '__main__':
    seed_everything()
    main()
