import os
import sys
sys.path.append("..")
sys.path.append("/home/sci/blakez/code/")
import GenerateDeformation as gd
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

    output_grid = io.LoadITKFile(
        '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/011_----_3D_VIBE_0p5iso_cor_3ave.nii.gz',
        device=device
    )

    aff_grid = core.StructuredGrid.FromGrid(output_grid, channels=3)
    del output_grid
    torch.cuda.empty_cache()
    aff_grid.set_size((256, 256, 256), inplace=True)
    aff_grid.set_to_identity_lut_()

    # Load the affine
    aff = np.loadtxt(f'{day3_dir}surfaces/raw/exvivo_to_invivo_affine.txt')
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
        f'{day3_dir}volumes/deformable/invivo_phi.mhd', device=device
    )
    phi.set_size((256, 256, 256))

    deformation = so.ComposeGrids.Create(device=device)([aff_grid, phi])

    return deformation


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


def mr_to_block(block_path, rabbit, t1_vol, t2_vol, ext_vols=[]):

    day3_dir = f'/hdscratch/ucair/{rabbit}/mri/invivo/'
    stacked_blocks_dir = f'/hdscratch/ucair/{rabbit}/blockface/recons/'
    block = block_path.split('/')[-1]
    exvivo_out = f'/hdscratch/ucair/{rabbit}/mri/exvivo/volumes/deformable/{block}/'
    invivo_out = f'/hdscratch/ucair/{rabbit}/mri/invivo/volumes/deformable/{block}/'
    if not os.path.exists(invivo_out):
        os.makedirs(invivo_out)

    # deformations = []

    # # Get the transformation with the affine included from invivo to exvivo
    # invivo_to_exvivo = mr_to_exvivo(day3_dir)
    # deformations.append(invivo_to_exvivo)
    #
    # # Get the transformation with the affine included from exvivo to the stacked blocks
    # exvivo_to_stacked = exvivo_to_blocks(stacked_blocks_dir)
    # deformations.append(exvivo_to_stacked)
    #
    # # Load the transformation from the stacked blocks to the block of interest
    # stacked_to_block = stacked_blocks_to_block(block_path)
    # deformations.append(stacked_to_block)

    # # Load the transformation for stitching if there is one
    # if os.path.exists(f'{block_path}/volumes/raw/{block}_phi_stitch.mhd'):
    #     block_stitching = block_stitch(block_path)
    #     deformations.append(block_stitching)

    # Genereate the deformation for the block
    to_block = gd.generate(rabbit, block=block, source_space='invivo', target_space='blockface')

    # Load the Day3 MR file to be deformed
    t2_motion = io.LoadITKFile(t2_vol, device=device)
    ce_t1 = io.LoadITKFile(t1_vol, device=device)
    adc = io.LoadITKFile(ext_vols[0], device=device)

    torch.cuda.empty_cache()

    t2_to_block = so.ApplyGrid.Create(to_block, device=device, pad_mode='zeros')(t2_motion, to_block)
    io.SaveITKFile(t2_to_block, f'{invivo_out}invivo_t2_to_{block}.mhd')

    t1_to_block = so.ApplyGrid.Create(to_block, device=device, pad_mode='zeros')(ce_t1, to_block)
    io.SaveITKFile(t1_to_block, f'{invivo_out}invivo_ce_t1_to_{block}.mhd')
    adc_to_block = so.ApplyGrid.Create(to_block, device=device, pad_mode='zeros')(adc, to_block)
    io.SaveITKFile(adc_to_block, f'{invivo_out}invivo_adc_to_{block}.mhd')

    # # Load the exvivo
    # ex_ce_t1 = io.LoadITKFile(
    #     '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/011_----_3D_VIBE_0p5iso_cor_3ave.nii.gz',
    #     device=device
    # )
    #
    # ex_t2 = io.LoadITKFile(
    #     '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/015_----_t2_spc_0p5x0p5x1mm_iso_satBand_3p5ave'
    #     '.nii.gz',
    #     device=device
    # )

    # exvivo_to_block = so.ComposeGrids.Create(device=device)(deformations[1:][::-1])
    # t1_ex_to_block = so.ApplyGrid.Create(exvivo_to_block, device=device, pad_mode='zeros')(ex_ce_t1, exvivo_to_block)
    # io.SaveITKFile(t1_ex_to_block, f'{exvivo_out}exvivo_ce_t1_to_{block}.mhd')
    #
    # t2_ex_to_block = so.ApplyGrid.Create(exvivo_to_block, device=device, pad_mode='zeros')(ex_t2, exvivo_to_block)
    # io.SaveITKFile(t2_ex_to_block, f'{exvivo_out}exvivo_t2_to_{block}.mhd')

    print(f'Done with {block}')


if __name__ == '__main__':
    rabbit = '18_062'
    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    t2_vol = '/hdscratch/ucair/18_062/mri/invivo/volumes/raw/016_m--e_t2_spc_1mm_iso_cor_TE300_2ave.nii.gz'
    adc_vol = '/hdscratch/ucair/18_062/mri/invivo/volumes/raw/024_m--e_2D_ss_DWI_ADC.nii.gz'
    t1_vol = '/hdscratch/ucair/18_062/mri/invivo/volumes/raw/029_----_3D_VIBE_0.5x0.5x1_NoGrappa_3avg_fatsat_cor.nii.gz'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    for block_path in [block_list[4]]:
        mr_to_block(block_path, rabbit, t2_vol=t2_vol, t1_vol=t1_vol, ext_vols=[adc_vol])
