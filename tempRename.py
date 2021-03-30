import os
import sys
sys.path.append("/home/sci/blakez/code/")
import glob
import yaml
import copy
import h5py
import shutil
import tools
import torch
import numpy as np
import subprocess as sp
import skimage.segmentation as seg

from collections import OrderedDict
import torch.optim as optim

import CAMP.camp.Core as core
import CAMP.camp.FileIO as io
import CAMP.camp.StructuredGridTools as st
import CAMP.camp.UnstructuredGridOperators as uo
import CAMP.camp.StructuredGridOperators as so

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'

block_list = sorted(glob.glob('/hdscratch/ucair/18_047/blockface/*'))

# Generate a hd_label_block for each block
for block_path in block_list[10:11]:
    block = block_path.split('/')[-1]
    if not os.path.exists(f'{block_path}/volumes/raw/hd_labels/'):
        os.makedirs(f'{block_path}/volumes/raw/hd_labels/')

    # Load the segmentation volume
    seg_vol = io.LoadITKFile(f'{block_path}/volumes/raw/segmentation_volume.mhd')

    hd_vol = seg_vol.copy()
    hd_vol *= 0.0

    raw_images = sorted(glob.glob(f'/hdscratch/ucair/18_047/microscopic/{block}/raw/*_image.tif'))
    cur_nums = [int(x.split('/')[-1].split('_')[1]) for x in raw_images]

    for num in cur_nums:
        hd_vol.data[:, num - 1] = seg_vol.data[:, num - 1].clone()

    io.SaveITKFile(hd_vol, f'{block_path}/volumes/raw/hd_labels/{block}_hdlabel_volume.mhd')

# for block_path in block_list[10:11]:
#     block = block_path.split('/')[-1]
#
#     # get the current image numbers
#     raw_images = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))
#     cur_nums = [int(x.split('/')[-1].split('_')[1]) for x in raw_images]
#
#     in_s = input(f'Input the actual blockface image numbers in order for {block}: ')
#     real_nums = list(map(int, in_s.split(',')))
#
#     if len(real_nums) != len(cur_nums):
#         print(f'There is a length discrepency: {len(real_nums)} != {len(cur_nums)}')
#         exit
#
#     for r, c in zip(real_nums, cur_nums):
#
#         #hdf5 image
#         i_name = f'{block_path}/hdf5/{block}_img{c:03d}_image.hdf5'
#         o_name = f'{block_path}/hdf5/{block}_img{r:03d}_image.hdf5'
#
#         try:
#             shutil.move(i_name, o_name)
#         except FileNotFoundError:
#             pass
#
#         #hdf5 label
#         i_name = f'{block_path}/hdf5/{block}_img{c:03d}_label.hdf5'
#         o_name = f'{block_path}/hdf5/{block}_img{r:03d}_label.hdf5'
#
#         try:
#             shutil.move(i_name, o_name)
#         except FileNotFoundError:
#             pass
#
#         #raw image
#         i_name = f'{block_path}/raw/IMG_{c:03d}_histopathology_image.tif'
#         o_name = f'{block_path}/raw/IMG_{r:03d}_histopathology_image.tif'
#
#         try:
#             shutil.move(i_name, o_name)
#         except FileNotFoundError:
#             pass
#
#         #raw label
#         i_name = f'{block_path}/raw/IMG_{c:03d}_histopathology_label.tif'
#         o_name = f'{block_path}/raw/IMG_{r:03d}_histopathology_label.tif'
#
#         try:
#             shutil.move(i_name, o_name)
#         except FileNotFoundError:
#             pass
#
#         # Move the segmentation directory
#         i_dir = f'{block_path}/segmentations/IMG_{c:03d}/'
#         o_dir = f'{block_path}/segmentations/IMG_{r:03d}/'
#
#         try:
#             shutil.move(i_dir, o_dir)
#         except FileNotFoundError:
#             pass
#
#         for file in sorted(glob.glob(f'{block_path}/segmentations/IMG_{r:03d}/*{c:03d}*')):
#
#             o_name = file.replace(f'{c:03d}', f'{r:03d}')
#             try:
#                 shutil.move(file, o_name)
#             except FileNotFoundError:
#                 pass
