import os
import sys
import glob
import copy
import torch
import numpy as np
import subprocess as sp

from collections import OrderedDict
import torch.optim as optim

from CAMP.Core import *
import CAMP.FileIO as io
from CAMP.UnstructuredGridOperators import *
from CAMP.StructuredGridOperators import ResampleWorld, ApplyGrid

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


device = 'cuda:1'
