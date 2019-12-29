# -*- coding: utf-8 -*-
from .resnet import *

select_model = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

def create(name, *args, **kwargs):
    if name not in select_model:
        raise KeyError("Unknown model:", name)
    return select_model[name](*args, **kwargs)
