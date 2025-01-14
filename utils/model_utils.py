import csv

import torch
from torch import nn
from thop import profile, clever_format
from torchinfo import summary
from ptflops import get_model_complexity_info
import os


def count_parameters(model):
    params = torch.tensor([p.numel() for p in model.parameters() if p.requires_grad]).sum()
    return params


def init_weights(m):
    if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear, nn.modules.conv._ConvTransposeNd)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def param_and_op_count(model, input_shape, out_dir=None, fname='op_count.csv', use_thop=True, use_ptflops=True,
                       use_torchinfo=True):
    input = torch.zeros(input_shape, device=next(model.parameters()).device)

    if use_thop:
        macs_thop, params_thop = profile(model, (input, ))
    if use_torchinfo:
        stats_torchinfo = summary(model, input_size=input_shape)
    if use_ptflops:
        macs_ptflops, params_ptflops = get_model_complexity_info(model, input_shape[1:], as_strings=False,
                                                                 print_per_layer_stat=False, verbose=True)

    if out_dir is not None:
        with open(os.path.join(out_dir, fname), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Counting method', 'Parameters', 'MACs'])
            if use_thop:
                writer.writerow(['thop', params_thop, macs_thop])
            if use_torchinfo:
                writer.writerow(['torchinfo', stats_torchinfo.trainable_params, stats_torchinfo.total_mult_adds])
            if use_ptflops:
                writer.writerow(['ptflops', params_ptflops, macs_ptflops])

    macs_thop, params_thop = clever_format([macs_thop, params_thop], "%.3f")
    print(macs_thop, params_thop)
    return macs_thop, params_thop
