import os
import sys
import glob
import yaml
import copy
import h5py
import tools
import torch
import numpy as np
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

device = 'cuda:1'


def stitch_surfaces(rabbit):

    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_ext = '/surfaces/raw/'
    vol_ext = '/volumes/raw/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    rerun = True
    skip_blocks = ['block06', 'block07']
    # skip_blocks = ['block08', 'block05', 'block09', 'block10', 'block11', 'block12']
    # skip_blocks = ['block07', 'block08', 'block09']

    for i, block_path in enumerate(block_list):

        block = block_path.split('/')[-1]
        target_surface_paths = sorted(glob.glob(f'{rabbit_dir}{block}{raw_ext}{block}_target_piece_surface*.obj'))
        source_surface_paths = sorted(glob.glob(f'{rabbit_dir}{block}{raw_ext}{block}_source_piece_surface*.obj'))

        target_surface_paths = [x for x in target_surface_paths if 'stitched' not in x]
        source_surface_paths = [x for x in source_surface_paths if 'stitched' not in x]

        if block in skip_blocks:
            print(f'Skipping {block} ... ')
            continue

        if target_surface_paths == [] and source_surface_paths == []:
            print(f'No stitching surfaces for {block} ... ')
            continue

        for target_surface_path, source_surface_path in zip(target_surface_paths, source_surface_paths):

            try:
                verts, faces = io.ReadOBJ(target_surface_path)
                tar_surface = core.TriangleMesh(verts, faces)
                tar_surface.to_(device)
            except IOError:
                print(f'The target stitching surface for {block} was not found ... skipping')
                continue

            try:
                verts, faces = io.ReadOBJ(source_surface_path)
                src_surface = core.TriangleMesh(verts, faces)
                src_surface.to_(device)
                src_surface.flip_normals_()
            except IOError:
                print(f'The source stitching surface for {block} was not found ... skipping')
                continue

            piece_ext = target_surface_path.split('_')[-1].split('.')[0]

            try:
                with open(f'{rabbit_dir}{block}{raw_ext}{block}_stitch_config_{piece_ext}.yaml', 'r') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)
            except IOError:
                params = {
                    'spatial_sigma': [1.0],
                    'smoothing_sigma': [150.0, 150.0, 150.0],
                    'deformable_lr': [0.0001],
                    'converge': 0.05,
                    'rigid_transform': True,
                    'phi_inv_size': [25, 128, 128]
                }

            # Do the deformable registration
            def_src_surface = tools.deformable_register_no_phi(
                tar_surface.copy(),
                src_surface.copy(),
                deformable_lr=params['deformable_lr'],
                spatial_sigma=params['spatial_sigma'],
                smoothing_sigma=params['smoothing_sigma'],
                converge=params['converge'],
                device=device
            )

            new_verts = src_surface.vertices.clone() + ((def_src_surface.vertices - src_surface.vertices) * 0.5)
            mid_surface = src_surface.copy()
            mid_surface.vertices = new_verts.clone()
            mid_surface.calc_normals()
            mid_surface.calc_centers()

            # Load the target ext
            try:
                verts, faces = io.ReadOBJ(f'{rabbit_dir}{block}{raw_ext}{block}_decimate.obj')
                surface_ext = core.TriangleMesh(verts, faces)
                surface_ext.to_(device)
            except IOError:
                print(f'The target exterior surface for {block} was not found')


            try:
                with open(f'{rabbit_dir}{block}{raw_ext}{block}_stitch_config_{piece_ext}.yaml', 'r') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)
            except IOError:
                params = {
                    'spatial_sigma': [0.5],
                    'smoothing_sigma': [150.0, 150.0, 150.0],
                    'deformable_lr': [0.00001],
                    'converge': 1.0,
                    'rigid_transform': True,
                    'phi_inv_size': [25, 25, 25]
                }
            # Do the deformable registration
            def_src_surface, def_tar_surface, def_src_excess, phi_inv = tools.stitch_surfaces(
                tar_surface.copy(),
                src_surface.copy(),
                mid_surface.copy(),
                deformable_lr=params['deformable_lr'],
                spatial_sigma=params['spatial_sigma'],
                smoothing_sigma=params['smoothing_sigma'],
                phi_inv_size=params['phi_inv_size'],
                converge=params['converge'],
                src_excess=[surface_ext],
                device=device,
            )

            print('Done')

            # Save out the parameters:
            # with open(f'{rabbit_dir}{block}{raw_ext}{block}_stitch_config_{piece_ext}.yaml', 'w') as f:
            #     yaml.dump(params, f)

            # Save out all of the deformable transformed surfaces and phi inv
            # io.SaveITKFile(phi_inv, f'{rabbit_dir}{block}{vol_ext}{block}_phi_inv_stitch__{piece_ext}.mhd')
            # out_path = f'{rabbit_dir}{block}{raw_ext}{block}'
            # io.WriteOBJ(def_src_surface.vertices, def_src_surface.indices,
            #             f'{out_path}_source_piece_surface_{piece_ext}_stitched.obj')

            # for extra_path, extra_surface in zip(extras_paths, def_extras):
            #     name = extra_path.split('/')[-1].split(f'{block}')[-1].replace('.', '_stitched.')
            #     if not os.path.exists(f'{out_path}{name}') or rerun:
            #         if 'decimate' in name:
            #             continue
            #         io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')


if __name__ == '__main__':
    rabbit = '18_047'
    stitch_surfaces(rabbit)
