# from .RabbitCommon import *
import os
import glob
import yaml
import copy
import h5py
import tools
import torch
import shutil
import numpy as np
from torch.autograd import Variable
from CAMP.UnstructuredGridOperators import *
import subprocess as sp
import skimage.segmentation as seg

from collections import OrderedDict
import torch.optim as optim

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridTools as st
import CAMP.UnstructuredGridOperators as uo
import CAMP.StructuredGridOperators as so

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


def main(rabbit):
    import RabbitCommon as rc

    in_path = f'/hdscratch/rabbit_data/{rabbit}/rawDicoms/ExVivo*/*'
    out_path = f'/hdscratch/ucair/{rabbit}/mri/exvivo/volumes/raw/'
    files = sorted(glob.glob(in_path))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for cnt, dcm_file in enumerate(files):
        print(f'Processing {dcm_file} ... ', end='')

        out_name = dcm_file.split('/')[-1].split(' ')[-1]
        if os.path.exists(f'{out_path}{cnt:02d}_{out_name}.nii.gz'):
            print('done')
            continue
        vol = rc.LoadDICOM(dcm_file, 'cpu')
        io.SaveITKFile(vol, f'{out_path}{cnt:02d}_{out_name}.nii.gz')
        print('done')

    # Copy over some of the 'invivo' (day3)
    out_path = f'/hdscratch/ucair/{rabbit}/mri/invivo/volumes/raw/'
    in_path = f'/hdscratch/rabbit_data/{rabbit}/rawVolumes/Post*/*'
    files = sorted(glob.glob(in_path))
    filt_files = [x for x in files if '3D_VIBE_0.5' in x]
    filt_files += [x for x in files if 't2' in x]
    filt_files += [x for x in files if 'Day3' in x]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for f in filt_files:
        shutil.copy(f, f'{out_path}{f.split("/")[-1]}')

    # Copy over some of the day0
    out_path = f'/hdscratch/ucair/{rabbit}/mri/day0/volumes/raw/'
    in_path = f'/hdscratch/rabbit_data/{rabbit}/rawVolumes/Ablation*/*'
    files = sorted(glob.glob(in_path))
    filt_files = [x for x in files if '3D_VIBE_0.5' in x]
    filt_files += [x for x in files if 't2' in x]
    filt_files += [x for x in files if 'Day0' in x]

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for f in filt_files:
        shutil.copy(f, f'{out_path}{f.split("/")[-1]}')


if __name__ == '__main__':
    rabbit = '18_062'
    # process_mic(rabbit)
    # match_bf_mic()
    main(rabbit)
