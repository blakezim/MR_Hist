import os
import sys
sys.path.append("/home/sci/blakez/code/")
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


def mr_to_exvivo(rabbit, block, direction, affine_only=False):

    invivo_dir = f'/hdscratch/ucair/{rabbit}/mri/invivo/'
    exvivo_dir = f'/hdscratch/ucair/{rabbit}/mri/exvivo/'

    # Load the affine
    aff = np.loadtxt(f'{exvivo_dir}surfaces/raw/exvivo_to_invivo_affine.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    if affine_only:
        if direction == 'phi':
            return aff.inverse()
        else:
            return aff

    # Load the deformation
    deformation = io.LoadITKFile(
        f'{exvivo_dir}volumes/raw/exvivo_to_invivo_{direction}.mhd', device=device
    )
    deformation.set_size((256, 256, 256))

    if direction == 'phi':
        output_grid = io.LoadITKFile(f'{exvivo_dir}volumes/raw/reference_volume.nii.gz', device=device)
        aff_grid = core.StructuredGrid.FromGrid(output_grid, channels=3)
        del output_grid
        torch.cuda.empty_cache()
        aff_grid.set_size((256, 256, 256), inplace=True)
        aff_grid.set_to_identity_lut_()

        # Apply the FORWARD affine to the deformation
        a = aff[0:3, 0:3].float()
        t = aff[-0:3, 3].float()

        # Create a deformation from the affine that lives in the stacked blocks space
        aff_grid.data = aff_grid.data.flip(0)
        aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        aff_grid.data = (aff_grid.data.squeeze() + t).permute([-1] + list(range(0, 3)))
        aff_grid.data = aff_grid.data.flip(0)

        invivo_exvivo = so.ComposeGrids.Create(device=device)([aff_grid, deformation])

    else:

        deformation.data = deformation.data.flip(0)

        # Apply the inverse affine to the grid
        aff = aff.inverse()
        a = aff[0:3, 0:3].float()
        t = aff[-0:3, 3].float()

        deformation.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                        deformation.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        deformation.data = (deformation.data.squeeze() + t).permute([-1] + list(range(0, 3)))

        # Flip phi_inv back to the way it was
        deformation.data = deformation.data.flip(0)

        invivo_exvivo = deformation.copy()

    return invivo_exvivo


def exvivo_to_stacked_blocks(rabbit, block, direction, affine_only=False):

    # This is registered from blocks to exvivo, so phi is needed to bring the exvivo MR image to the block images
    # Need to determine the grid to sample the MR onto
    # rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    block_dir = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'
    exvivo_dir = f'/hdscratch/ucair/{rabbit}/mri/exvivo/'
    # block_list = sorted(glob.glob(f'{rabbit_dir}block*'))
    # orig_dir = f'/home/sci/blakez/ucair/{rabbit}/rawVolumes/ExVivo_2018-07-26/'

    # Load the affine
    try:
        aff = np.loadtxt(f'{exvivo_dir}/surfaces/raw/blocks_to_exvivo_affine.txt')
        aff = torch.tensor(aff, device=device, dtype=torch.float32)
    except IOError:
        aff = np.loadtxt(f'{block_dir}../recons/surfaces/raw/blocks_to_exvivo_affine.txt')
        aff = torch.tensor(aff, device=device, dtype=torch.float32)


    if affine_only:
        if direction == 'phi':
            return aff.inverse()
        else:
            return aff

    # Load the deformation
    try:
        deformation = io.LoadITKFile(
            f'{exvivo_dir}/volumes/raw/blocks_{direction}_to_exvivo.mhd', device=device
        )
        deformation.set_size((256, 256, 256))
    except RuntimeError:
        deformation = io.LoadITKFile(
            f'{block_dir}../recons/surfaces/raw/blocks_{direction}_to_exvivo.mhd', device=device
        )
        deformation.set_size((256, 256, 256))

    if direction == 'phi':

        spacing = []
        origin = []
        size = []

        # if 'block07' in block_path:
        #     hdr = tools.read_mhd_header(f'{block_path}/volumes/raw/difference_volume.mhd')
        # else:
        hdr = tools.read_mhd_header(f'{block_dir}/volumes/raw/{block}_phi_inv_stacking.mhd')
        spacing.append(np.array([float(x) for x in hdr['ElementSpacing'].split(' ')]))
        origin.append(np.array([float(x) for x in hdr['Offset'].split(' ')]))
        size.append(np.array([float(x) for x in hdr['DimSize'].split(' ')]))

        spacing = np.stack(spacing)
        origin = np.stack(origin)
        size = np.stack(size)

        extent = size * spacing + origin
        aff_grid_size = np.array((512, 512, 512))
        aff_grid_origin = np.min(origin, axis=0)
        aff_grid_spacing = (np.max(extent, axis=0) - aff_grid_origin) / aff_grid_size

        aff_grid = core.StructuredGrid(
            size=aff_grid_size.tolist()[::-1],
            spacing=aff_grid_spacing.tolist()[::-1],
            origin=aff_grid_origin.tolist()[::-1],
            device=device,
            channels=3
        )

        aff_grid.set_size(size=(512, 512, 512), inplace=True)
        aff_grid.set_to_identity_lut_()

        # Apply the FORWARD affine to the deformation
        a = aff[0:3, 0:3].float()
        t = aff[-0:3, 3].float()

        # Create a deformation from the affine that lives in the stacked blocks space
        aff_grid.data = aff_grid.data.flip(0)
        aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        aff_grid.data = (aff_grid.data.squeeze() + t).permute([-1] + list(range(0, 3)))
        aff_grid.data = aff_grid.data.flip(0)

        # Compose the grids
        exvivo_to_blocks = so.ComposeGrids.Create(device=device)([aff_grid, deformation])

    else:

        deformation.data = deformation.data.flip(0)

        # Apply the inverse affine to the grid
        aff = aff.inverse()
        a = aff[0:3, 0:3].float()
        t = aff[-0:3, 3].float()

        deformation.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                        deformation.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        deformation.data = (deformation.data.squeeze() + t).permute([-1] + list(range(0, 3)))

        # Flip phi_inv back to the way it was
        deformation.data = deformation.data.flip(0)

        exvivo_to_blocks = deformation.copy()

    return exvivo_to_blocks


def stacked_blocks_to_block(rabbit, block, direction, affine_only=False):

    block_dir = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'

    # Load the affine
    aff = np.loadtxt(f'{block_dir}/surfaces/raw/{block}_rigid_tform.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    if affine_only:
        if direction == 'phi':
            return aff.inverse()
        else:
            return aff

    # Load the deformabale transformation
    deformation = io.LoadITKFile(
        f'{block_dir}/volumes/raw/{block}_{direction}_stacking.mhd', device=device
    )
    deformation.set_size((60, 1024, 1024))

    if direction == 'phi':
        # Need to determine the output grid
        output_grid = io.LoadITKFile(f'{block_dir}/volumes/raw/difference_volume.mhd', device=device)
        aff_grid = core.StructuredGrid.FromGrid(output_grid, channels=3)
        del output_grid
        torch.cuda.empty_cache()
        aff_grid.set_size((60, 1024, 1024), inplace=True)
        aff_grid.set_to_identity_lut_()

        # Apply the FORWARD affine to the deformation
        a = aff[0:3, 0:3].float()
        t = aff[-0:3, 3].float()

        # Create a deformation from the affine that lives in the stacked blocks space
        aff_grid.data = aff_grid.data.flip(0)
        aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        aff_grid.data = (aff_grid.data.squeeze() + t).permute([-1] + list(range(0, 3)))
        aff_grid.data = aff_grid.data.flip(0)

        # Compose the grids
        stackblocks_to_block = so.ComposeGrids.Create(device=device)([aff_grid, deformation])

    else:

        deformation.data = deformation.data.flip(0)

        # Apply the inverse affine to the grid
        aff = aff.inverse()
        a = aff[0:3, 0:3].float()
        t = aff[-0:3, 3].float()

        deformation.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                        deformation.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        deformation.data = (deformation.data.squeeze() + t).permute([-1] + list(range(0, 3)))

        # Flip phi_inv back to the way it was
        deformation.data = deformation.data.flip(0)

        stackblocks_to_block = deformation.copy()

    return stackblocks_to_block


def block_stitch(rabbit, block, direction):

    # block = block_path.split('/')[-1]
    block_path = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'

    # Load the deformabale transformation
    stitch_block = io.LoadITKFile(
        f'{block_path}/volumes/raw/{block}_{direction}_stitch.mhd', device=device
    )
    stitch_block.set_size((60, 1024, 1024))

    return stitch_block


def generate_affine_only(rabbit, block=None, img_num=None, source_space='invivo', target_space='blockface'):
    block_spaces = ['stacked', 'blockface']
    mri_spaces = ['invivo', 'exvivo']
    hist_spaces = ['hist']

    space_list = mri_spaces + block_spaces + hist_spaces

    if (source_space in block_spaces or target_space in block_spaces) and block == None:
        raise Exception(f'You must specify a block number for a deformation from {source_space} to {target_space}.')

    if (source_space in hist_spaces or target_space in hist_spaces) and img_num == None:
        raise Exception(f'You must specify an image number for a deformation from {source_space} to {target_space}.')

    source_idx = space_list.index(source_space)
    target_idx = space_list.index(target_space)
    function_list = [mr_to_exvivo, exvivo_to_stacked_blocks, stacked_blocks_to_block]

    if target_idx > source_idx:
        def_dir = 'phi'

        affines = []
        for d in range(source_idx, target_idx):
            affines.append(function_list[d](
                rabbit, block, def_dir, affine_only=True
            ))

        if len(affines) == 1:
            out_affine = affines[0].clone()
        else:
            out_affine = affines[0].clone()
            for aff in affines[1:]:
                out_affine = torch.matmul(aff, out_affine)

    else:
        def_dir = 'phi_inv'

        affines = []
        for d in range(target_idx, source_idx):
            affines.append(function_list[d](
                rabbit, block, def_dir, affine_only=True
            ))

        if len(affines) == 1:
            out_affine = affines[0].clone()
        else:
            affines = affines[::-1]
            out_affine = affines[0].clone()
            for aff in affines[1:]:
                out_affine = torch.matmul(aff, out_affine)

    return out_affine


def generate(rabbit, block=None, img_num=None, source_space='invivo', target_space='blockface', stitch=True):

    block_spaces = ['stacked', 'blockface']
    mri_spaces = ['invivo', 'exvivo']
    hist_spaces = ['hist']

    space_list = mri_spaces + block_spaces + hist_spaces

    if (source_space in block_spaces or target_space in block_spaces) and block == None:
        raise Exception(f'You must specify a block number for a deformation from {source_space} to {target_space}.')

    if (source_space in hist_spaces or target_space in hist_spaces) and img_num == None:
        raise Exception(f'You must specify an image number for a deformation from {source_space} to {target_space}.')

    source_idx = space_list.index(source_space)
    target_idx = space_list.index(target_space)
    function_list = [mr_to_exvivo, exvivo_to_stacked_blocks, stacked_blocks_to_block]

    if target_idx > source_idx:
        def_dir = 'phi'

        deformations = []
        for d in range(source_idx, target_idx):
            deformations.append(function_list[d](
                rabbit, block, def_dir
            ))

        if stitch and os.path.exists(
                f'/hdscratch/ucair/{rabbit}/blockface/{block}/volumes/raw/{block}_{def_dir}_stitch.mhd'):
            deformations.append(block_stitch(rabbit, block, def_dir))

        full_deformation = so.ComposeGrids.Create(device=device)(deformations[::-1])

    else:
        def_dir = 'phi_inv'

        deformations = []
        for d in range(target_idx, source_idx):
            deformations.append(function_list[d](
                rabbit, block, def_dir
            ))

        if stitch and os.path.exists(
                f'/hdscratch/ucair/{rabbit}/blockface/{block}/volumes/raw/{block}_{def_dir}_stitch.mhd'):
            deformations.append(block_stitch(rabbit, block, def_dir))

        full_deformation = so.ComposeGrids.Create(device=device)(deformations)

    return full_deformation


if __name__ == '__main__':
    rabbit = '18_047'
    block = 'block07'
    generate(rabbit, block=block)
