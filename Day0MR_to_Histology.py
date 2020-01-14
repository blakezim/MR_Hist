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


def mr_to_exvivo(day3_dir):

    # This is registered from invivo to exvivo, so phi_inv is needed to bring the invivo MR image to the exvivo images
    # Load the affine
    aff = np.loadtxt(f'{day3_dir}surfaces/raw/invivo_to_exvivo_affine.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    # Load the defromabale transformation
    phi_inv = io.LoadITKFile(
        f'{day3_dir}volumes/deformable/invivo_phi_inv.mhd', device=device
    )
    phi_inv.set_size((256, 256, 256))
    phi_inv.data = phi_inv.data.flip(0)

    # Apply the inverse affine to the deformation
    aff = aff.inverse()
    a = aff[0:3, 0:3].float()
    t = aff[-0:3, 3].float()

    phi_inv.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                phi_inv.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    phi_inv.data = (phi_inv.data.squeeze() + t).permute([-1] + list(range(0, 3)))

    # Flip phi_inv back to the way it was
    phi_inv.data = phi_inv.data.flip(0)

    return phi_inv


def exvivo_to_blocks(stacked_blocks_dir):

    # This is registered from blocks to exvivo, so phi is needed to bring the exvivo MR image to the block images
    # Need to determine the grid to sample the MR onto
    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))
    orig_dir = f'/home/sci/blakez/ucair/{rabbit}/rawVolumes/ExVivo_2018-07-26/'
    spacing = []
    origin = []
    size = []
    for block_path in block_list:

        if 'block07' in block_path:
            hdr = tools.read_mhd_header(f'{block_path}/volumes/raw/difference_volume.mhd')
        else:
            hdr = tools.read_mhd_header(f'{block_path}/volumes/deformable/difference_volume_deformable.mhd')
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

    # Load the affine
    aff = np.loadtxt(f'{orig_dir}blocks_to_exvivo_affine.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    # Apply the FORWARD affine to the deformation
    # aff = aff.inverse()
    a = aff[0:3, 0:3].float()
    t = aff[-0:3, 3].float()

    # Create a deformation from the affine that lives in the stacked blocks space

    aff_grid.data = aff_grid.data.flip(0)

    aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    aff_grid.data = (aff_grid.data.squeeze() + t).permute([-1] + list(range(0, 3)))

    aff_grid.data = aff_grid.data.flip(0)

    # Load the defromabale transformation
    phi = io.LoadITKFile(
        f'{orig_dir}blocks_phi_to_exvivo.mhd', device=device
    )
    phi.set_size((256, 256, 256))

    # Compose the grids
    deformation = so.ComposeGrids.Create(device=device)([aff_grid, phi])

    return deformation


def stacked_blocks_to_block(block_path):

    block = block_path.split('/')[-1]

    # Need to determine the output grid
    output_grid = io.LoadITKFile(f'{block_path}/volumes/raw/difference_volume.mhd', device=device)
    aff_grid = core.StructuredGrid.FromGrid(output_grid, channels=3)
    del output_grid
    torch.cuda.empty_cache()
    aff_grid.set_size((60, 1024, 1024), inplace=True)
    aff_grid.set_to_identity_lut_()

    # Load the affine
    aff = np.loadtxt(f'{block_path}/surfaces/raw/{block}_rigid_tform.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    # Apply the FORWARD affine to the deformation
    # aff = aff.inverse()
    a = aff[0:3, 0:3].float()
    t = aff[-0:3, 3].float()

    # Create a deformation from the affine that lives in the stacked blocks space

    aff_grid.data = aff_grid.data.flip(0)

    aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    aff_grid.data = (aff_grid.data.squeeze() + t).permute([-1] + list(range(0, 3)))

    aff_grid.data = aff_grid.data.flip(0)

    # Load the deformabale transformation
    phi = io.LoadITKFile(
        f'{block_path}/volumes/raw/{block}_phi_stacking.mhd', device=device
    )
    phi.set_size((60, 1024, 1024))

    # Compose the grids
    deformation = so.ComposeGrids.Create(device=device)([aff_grid, phi])

    return deformation


def block_stitch(block_path):
    block = block_path.split('/')[-1]

    # Load the deformabale transformation
    phi = io.LoadITKFile(
        f'{block_path}/volumes/raw/{block}_phi_stitch.mhd', device=device
    )
    phi.set_size((60, 1024, 1024))

    return phi


def get_day0_to_day3(data_path):

    # Load the deformabale transformation
    phi_inv = io.LoadITKFile(
        f'{data_path}/volumes/raw/day0_to_day3_phi_inv.mhd', device=device
    )
    phi_inv.set_size((256, 256, 256))

    return phi_inv


def mr_to_block(block_path, rabbit):

    day0_dir = f'/hdscratch/ucair/{rabbit}/mri/day0/'
    day3_dir = f'/hdscratch/ucair/{rabbit}/mri/invivo/'
    stacked_blocks_dir = f'/hdscratch/ucair/{rabbit}/blockface/recons/'
    block = block_path.split('/')[-1]
    # exvivo_out = f'/hdscratch/ucair/18_047/mri/exvivo/volumes/deformable/{block}/'
    invivo_out = f'/hdscratch/ucair/18_047/mri/day0/volumes/deformable/{block}/'

    deformations = []

    # Get the transformation from day0 to day3
    day0_to_day3 = get_day0_to_day3(day0_dir)
    deformations.append(day0_to_day3)

    # Get the transformation with the affine included from invivo to exvivo
    invivo_to_exvivo = mr_to_exvivo(day3_dir)
    deformations.append(invivo_to_exvivo)

    # Get the transformation with the affine included from exvivo to the stacked blocks
    exvivo_to_stacked = exvivo_to_blocks(stacked_blocks_dir)
    deformations.append(exvivo_to_stacked)

    # Load the transformation from the stacked blocks to the block of interest
    stacked_to_block = stacked_blocks_to_block(block_path)
    deformations.append(stacked_to_block)

    # # Load the transformation for stitching if there is one
    # if os.path.exists(f'{block_path}/volumes/raw/{block}_phi_stitch.mhd'):
    #     block_stitching = block_stitch(block_path)
    #     deformations.append(block_stitching)

    # Load the Day3 MR file to be deformed
    t2_motion = io.LoadITKFile(
        '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/089_----_t2_spc_1mm_iso_cor_post20min.nii.gz',
        device=device
    )

    ce_t1 = io.LoadITKFile(
        '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/102_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_00.nii.gz',
        device=device
    )

    ctd = io.LoadITKFile(
        '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/CTD_map.nrrd',
        device=device
    )
    ctd.data[ctd.data > 999999] = 999999
    ctd.data = torch.log(ctd.data + 0.001)
    ctd.data = (ctd.data - ctd.min()) / (ctd.max() - ctd.min())

    torch.cuda.empty_cache()

    invivo_to_block = so.ComposeGrids.Create(device=device)(deformations[::-1])
    t2_to_block = so.ApplyGrid.Create(invivo_to_block, device=device, pad_mode='zeros')(t2_motion, invivo_to_block)
    io.SaveITKFile(t2_to_block, f'{invivo_out}day0_t2_to_{block}.mhd')

    t1_to_block = so.ApplyGrid.Create(invivo_to_block, device=device, pad_mode='zeros')(ce_t1, invivo_to_block)
    io.SaveITKFile(t1_to_block, f'{invivo_out}day0_ce_t1_to_{block}.mhd')

    ctd_to_block = so.ApplyGrid.Create(invivo_to_block, device=device, pad_mode='zeros')(ctd, invivo_to_block)
    io.SaveITKFile(ctd_to_block, f'{invivo_out}day0_ctd_to_{block}.mhd')

    print(f'Done with {block}')


if __name__ == '__main__':
    rabbit = '18_047'
    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    for block_path in block_list[7:]:
        mr_to_block(block_path, rabbit)
