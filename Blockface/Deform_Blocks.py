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

device = 'cuda:1'


def blocks_to_exvivo(rabbit, base_dir='/hdscratch/ucair/'):
    # import RabbitCommon as rc

    base_path = f'{base_dir}{rabbit}/blockface/'
    block_list = sorted(glob.glob(f'{base_path}block*'))

    device = 'cuda:1'
    rerun = True

    # Deform all of the blocks
    for block_path in block_list:
        block = block_path.split('/')[-1]
        print(f'Deforming {block} ... ', end='')
        out_path = f'{base_path}{block}/'

        if not os.path.exists(f'{out_path}/volumes/deformable/'):
            os.makedirs(f'{out_path}/volumes/deformable/')

        bf_name = f'{block_path}/volumes/raw/scatter_volume.mhd'
        block_vol = io.LoadITKFile(bf_name, device=device)

        # generate the deformation to the stacked block
        if not os.path.exists(f'{out_path}/volumes/deformable/{block}_to_exvivo.nii.gz') or rerun:
            to_exvivo = generate(rabbit, block=block, source_space='blockface', target_space='exvivo', base_dir=base_dir)
            block_as_exvivo = so.ApplyGrid.Create(to_exvivo, device=device)(block_vol, to_exvivo)
            io.SaveITKFile(block_as_exvivo, f'{out_path}/volumes/deformable/{block}_to_exvivo.nii.gz')
            del block_as_exvivo, to_exvivo
            torch.cuda.empty_cache()

        print('done')


def deform_histology_volumes(rabbit, block):

    rerun = True

    blockface_dir = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'
    histology_dir = f'/hdscratch/ucair/{rabbit}/microscopic/{block}/'

    raw_dir = f'{histology_dir}/volume/raw/'
    out_dir = f'{histology_dir}/volume/deformable/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.exists(f'{out_dir}/{block}_ablation_segmentation_to_exvivo.mhd') and not rerun:
        return

    deformation = generate(rabbit, block, source_space='blockface', target_space='exvivo')
    deformation.to_(device=device)
    # Load the original volume
    ablation_vol = io.LoadITKFile(f'{raw_dir}/{block}_ablation_segmentation.mhd', device=device)
    ablation_vol.data = (ablation_vol.data >= 0.5).float()
    # combined_vol = io.LoadITKFile(f'{raw_dir}/{block}_ablation_and_transition_segmentation.mhd', device=device)
    def_ablation = so.ApplyGrid.Create(deformation, device=device)(ablation_vol, deformation)
    # def_combined = so.ApplyGrid.Create(deformation, device=device)(combined_vol, deformation)
    io.SaveITKFile(def_ablation, f'{out_dir}/{block}_ablation_segmentation_to_exvivo.mhd')
    # io.SaveITKFile(def_combined, f'{out_dir}/{block}_ablation_and_transition_segmentation_to_exvivo.mhd')


def stack_hitology_volumes(rabbit, block_paths):

    def_ablations = []
    def_combineds = []
    out_dir = f'/hdscratch/ucair/{rabbit}/microscopic/recons/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for block_path in block_paths:
        block = block_path.split('/')[-1]
        histology_dir = f'/hdscratch/ucair/{rabbit}/microscopic/{block}/'
        def_dir = f'{histology_dir}/volume/deformable/'

        # Load the deformed volume
        def_ablation = io.LoadITKFile(f'{def_dir}/{block}_ablation_segmentation_to_exvivo.mhd', device=device)
        # def_combined = io.LoadITKFile(f'{def_dir}/{block}_ablation_and_transition_segmentation_to_invivo.mhd',
        #                               device=device)

        # Threshold the volume
        def_ablation.data = (def_ablation.data >= 0.8).float()
        # def_combined.data = (def_combined.data >= 0.8).float()

        def_ablations.append(def_ablation)
        # def_combineds.append(def_combined)

    full_ablation = def_ablations[0].copy()
    full_ablation = full_ablation * 0.0
    for i, v in enumerate(def_ablations, 1):
        if v.sum() != 0.0:
            io.SaveITKFile(v, f'/home/sci/blakez/test_outs/volume{i}.nii.gz')
        full_ablation = full_ablation + v

    full_ablation.data = (full_ablation.data > 0.0).float()

    io.SaveITKFile(full_ablation, f'{out_dir}all_ablation_segs_to_exvivo.mhd')

    # full_combined = def_ablations[0].copy()
    # full_combined = full_combined * 0.0
    # for v in def_combineds:
    #     full_combined = full_combined + v
    #
    # full_combined.data = (full_combined.data > 0.0).float()

    # io.SaveITKFile(full_combined, f'{out_dir}all_ablation_and_transition_segs_to_invivo.mhd')


if __name__ == '__main__':
    rabbit = '18_047'
    blocks_to_exvivo(rabbit, base_dir='/hdscratch2/')
    # block_list = sorted(glob.glob(f'/hdscratch/ucair/{rabbit}/microscopic/block*'))
    # #
    # for block_path in block_list:
    #     block = block_path.split('/')[-1]
    #     deform_histology_volumes(rabbit, block)
    #
    # stack_hitology_volumes(rabbit, block_list)

