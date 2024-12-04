import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import *
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib
import skimage
import os
import re
import pathlib
import pickle 
data_gen = torch.Generator().manual_seed(4)
np.random.seed(4)


def assert_in_range(tensor, range, name='tensor'):
    assert len(range) == 2, 'range should be in form [min, max]'
    assert tensor.min() >= range[0], f'{name} should be in {range}, found: {tensor.min()}'
    assert tensor.max() <= range[1], f'{name} should be in {range}, found: {tensor.max()}'