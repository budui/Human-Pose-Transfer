from math import sqrt
import torch.nn as nn

__all__ = ["select"]

def select(name):
    registered_ways = {
        "tflib": weights_init_tflib,
        "normal": weights_init_normal,
        "xavier": weights_init_xavier,
        "": weights_init_torch
    }
    try:
        return registered_ways[name]
    except KeyError:
        raise RuntimeError("non-supported weights init way: {}".format(name))

def weights_init_torch(m):
    """do nothing"""
    pass

def weights_init_tflib(m):
    stdev = 0.02
    sqrt_v = sqrt(3)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.uniform_(m.weight.data, -stdev * sqrt_v, stdev * sqrt_v)
    elif classname.find('Linear') != -1:
        nn.init.uniform_(m.weight.data, -stdev * sqrt_v, stdev * sqrt_v)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)