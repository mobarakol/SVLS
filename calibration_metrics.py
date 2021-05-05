import torch
import numpy as np
import random
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

def ece_eval(preds, targets, n_bins=10, bg_cls = 0):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    confidences, predictions = confidences[targets>bg_cls], predictions[targets>bg_cls]
    accuracies = (predictions == targets[targets>bg_cls]) 
    Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0
    bin_idx = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        Bm[bin_idx] = bin_size
        if bin_size > 0:  
            accuracy_in_bin = np.sum(accuracies[in_bin])
            acc[bin_idx] = accuracy_in_bin / Bm[bin_idx]
            confidence_in_bin = np.sum(confidences[in_bin])
            conf[bin_idx] = confidence_in_bin / Bm[bin_idx]
        bin_idx += 1
        
    ece_all = Bm * np.abs((acc - conf))/ Bm.sum()
    ece = ece_all.sum() 
    return ece, acc, conf, Bm

def tace_eval(preds, targets, n_bins=10, threshold=1e-3, bg_cls = 0): #1e-3
    init = 0
    if bg_cls == 0:
        init = 1
    preds = preds.astype(np.float32)
    targets = targets.astype(np.float16)
    n_img, n_classes = preds.shape[:2]
    Bm_all, acc_all, conf_all = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    res = 0.0
    ece_all = []
    for cur_class in range(init, n_classes):
        cur_class_conf = preds[:, cur_class]
        cur_class_conf = cur_class_conf.flatten()
        cur_class_conf_sorted = np.sort(cur_class_conf)
        targets_vec = targets.flatten()
        targets_sorted = targets_vec[cur_class_conf.argsort()]
        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]
        bin_size = len(cur_class_conf_sorted) // n_bins
        ece_cls, Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
        bin_idx = 0  
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins-1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array

            Bm[bin_idx] = bin_size
            bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == cur_class)
            acc[bin_idx] = np.sum(bin_acc) / bin_size
            bin_conf = cur_class_conf_sorted[bin_start_ind : bin_end_ind]
            conf[bin_idx] = np.sum(bin_conf) / bin_size
            bin_idx += 1

        ece_cls = Bm * np.abs((acc - conf))/ Bm.sum()
        ece_all.append(np.mean(ece_cls))
        Bm_all += Bm
        acc_all += acc
        conf_all += conf
    ece, acc_all, conf_all = np.mean(ece_all),acc_all/(n_classes-init), conf_all/(n_classes-init)        
    return ece, acc_all, conf_all,Bm_all

def reliability_diagram(conf_avg, acc_avg, legend=None, leg_idx=0, n_bins=10):
    plt.figure(2)
    #plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot([conf_avg[acc_avg>0][0], 1], [conf_avg[acc_avg>0][0], 1], linestyle='--')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    #plt.xticks(np.arange(0, 1.1, 1/n_bins))
    #plt.title(title)
    plt.plot(conf_avg[acc_avg>0],acc_avg[acc_avg>0], marker='.', label = legend)
    plt.legend()
    plt.savefig('ece_reliability.png',dpi=300)