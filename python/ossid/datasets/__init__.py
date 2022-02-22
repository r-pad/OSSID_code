import os
import numpy as np
import torch

from torch.utils.data import DataLoader

def getDataloaders(cfg):
    if cfg.dataset.name == "fewshot_bop":
        from .fewshot_bop_dataset import getDataloaders as getDls
        return getDls(cfg)
    elif cfg.dataset.name == "render":
        from .render_dataset import getDataloaders as getDls
        return getDls(cfg)
    elif cfg.dataset.name == "dtoid":
        from .dtoid_dataset import getDataloaders as getDls
        return getDls(cfg)
    elif cfg.dataset.name == "dtoid_bop":
        from .dtoid_bop_dataset import getDataloaders as getDls
        return getDls(cfg)
    elif cfg.dataset.name == 'detect':
        from .detect_dataset import getDataloaders as getDls
        return getDls(cfg)
    else:
        raise Exception("Unknown cfg.dataset.name =", cfg.dataset.name)
