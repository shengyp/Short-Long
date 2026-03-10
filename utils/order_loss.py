import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 将真实标签转化为软标签
def true_metric_loss(true, no_of_classes, scale=1):
    batch_size = true.size(0)

    # 获取输入数据所在的设备 (是 cpu 还是 cuda)
    device = true.device

    true = true.view(batch_size, 1)

    # 在当前 device 上操作
    true_labels = true.long().repeat(1, no_of_classes).float()

    # 创建 class_labels 时指定 device
    class_labels = torch.arange(no_of_classes, device=device).float()

    phi = (scale * torch.abs(class_labels - true_labels))

    y = nn.Softmax(dim=1)(-phi)
    return y


def loss_function(output, labels, loss_type, expt_type=5, scale=1.8):
    targets = true_metric_loss(labels, expt_type, scale)
    return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()


def gr_metrics(op, t):
    op = np.array(op)
    t = np.array(t)
    TP = (op == t).sum()
    FN = (t > op).sum()
    FP = (t < op).sum()

    # 防止分母为0的保护措施
    if TP + FP == 0:
        GP = 0
    else:
        GP = TP / (TP + FP)

    if TP + FN == 0:
        GR = 0
    else:
        GR = TP / (TP + FN)

    if GP + GR == 0:
        FS = 0
    else:
        FS = 2 * GP * GR / (GP + GR)

    OE = (t - op > 1).sum()
    OE = OE / op.shape[0]

    return GP, GR, FS, OE