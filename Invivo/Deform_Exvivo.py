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
from CAMP.camp.UnstructuredGridOperators import *
import subprocess as sp
import skimage.segmentation as seg

from GenerateDeformation import generate, generate_affine_only

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


def main(rabbit, base_dir='/hdscratch/ucair/'):

    device = 'cuda:1'

    in_path = f'{base_dir}{rabbit}/mri/exvivo/volumes/raw/'
    out_base = f'{base_dir}{rabbit}/mri/exvivo/volumes/deformable/invivo/'
    if not os.path.exists(out_base):
        os.makedirs(out_base)

    to_invivo = generate(rabbit, source_space='exvivo', target_space='invivo', base_dir=base_dir)

    t2_file = sorted(glob.glob(f'{in_path}*t2*'))
    t1_file = sorted(glob.glob(f'{in_path}*VIBE*'))
    if len(t1_file) > 1:
        t1_file = [x for x in t1_file if '0p5' in x]

    exvivo_t2 = io.LoadITKFile(t2_file[0], device=device)
    exvivo_t1 = io.LoadITKFile(t1_file[0], device=device)

    def_t2 = so.ApplyGrid.Create(to_invivo, device=device)(exvivo_t2, to_invivo)
    def_t1 = so.ApplyGrid.Create(to_invivo, device=device)(exvivo_t1, to_invivo)

    io.SaveITKFile(def_t2, f'{out_base}exvivo_t2_def_to_invivo_{rabbit}.nii.gz')
    io.SaveITKFile(def_t1, f'{out_base}exvivo_t1_def_to_invivo_{rabbit}.nii.gz')
    # #
    # to_exvivo = generate(rabbit, source_space='invivo', target_space='exvivo')
    # to_exvivo_affine = generate_affine_only(rabbit, source_space='invivo', target_space='exvivo')
    #
    # invivo = io.LoadITKFile(sorted(glob.glob(f'/hdscratch/ucair/{rabbit}/mri/invivo/volumes/raw/*t2*'))[0], device=device)
    # aff_invivo = so.ApplyGrid.Create(to_exvivo_affine, device=device)(invivo, to_exvivo_affine)
    # def_invivo = so.ApplyGrid.Create(to_exvivo, device=device)(invivo, to_exvivo)
    #
    # io.SaveITKFile(aff_invivo, f'/home/sci/blakez/test_outs/invivo_aff_to_exvivo_npv_{rabbit}.nii.gz')
    # io.SaveITKFile(def_invivo, f'/home/sci/blakez/test_outs/invivo_def_to_exvivo_npv_{rabbit}.nii.gz')
    #
    # io.SaveITKFile(def_exvivo, f'{out_base}/invivo/{exvivo_name}_to_invivo.nii.gz')

    print('done')

    # # Deform all of the blocks
    # for block in block_list:
    #     print(f'Deforming {block} ... ', end='')
    #     out_path = f'{out_base}{block}/'
    #
    #     # if not os.path.exists(f'{out_path}/stacked/'):
    #     #     os.makedirs(f'{out_path}/stacked/')
    #     # if not os.path.exists(f'{out_path}/as_block/'):
    #     #     os.makedirs(f'{out_path}/as_block/')
    #
    #
    #
    #     # generate the deformation to the stacked block
    #     to_stacked = generate(rabbit, block=block, source_space='exvivo', target_space='stacked')
    #
    #
    #     # del stacked_exvivo, to_stacked
    #     # torch.cuda.empty_cache()
    #
    #     # generate to deformation to the raw block
    #     to_block = generate(rabbit, block=block, source_space='exvivo', target_space='blockface')
    #     block_exvivo = so.ApplyGrid.Create(to_block, device=device)(exvivo, to_block)
    #     io.SaveITKFile(block_exvivo, f'{out_path}/as_block/{exvivo_name}_to_{block}.nii.gz')


if __name__ == '__main__':
    rabbit_list = ['18_047', '18_060', '18_061', '18_062']

    main('18_062')
    # for rabbit in rabbit_list:
    #     if rabbit == '18_047':
    #         main(rabbit, base_dir='/hdscratch2/')
    #     else:
    #         main(rabbit)
