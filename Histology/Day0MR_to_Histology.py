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
import GenerateDeformation as gd
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


def mr_to_exvivo(day3_dir):

    output_grid = io.LoadITKFile(
        '/hdscratch/ucair/18_062/mri/exvivo/volumes/raw/04_3D_VIBE_0p5iso_cor_3ave.nii.gz',
        device=device
    )

    aff_grid = core.StructuredGrid.FromGrid(output_grid, channels=3)
    del output_grid
    torch.cuda.empty_cache()
    aff_grid.set_size((256, 256, 256), inplace=True)
    aff_grid.set_to_identity_lut_()

    # Load the affine
    aff = np.loadtxt(f'/hdscratch/ucair/18_062/mri/exvivo/surfaces/raw/exvivo_to_invivo_affine.txt')
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
        f'/hdscratch/ucair/18_062/mri/exvivo/volumes/raw/exvivo_to_invivo_phi.mhd', device=device
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
    invivo_out = f'/hdscratch/ucair/18_062/mri/day0/volumes/deformable/{block}/'
    if not os.path.exists(invivo_out):
        os.makedirs(invivo_out)

    deformations = []

    # # Get the transformation from day0 to day3
    # day0_to_day3 = get_day0_to_day3(day0_dir)
    # deformations.append(day0_to_day3)

    in_path = f'/hdscratch/ucair/{rabbit}/'

    # Need to load and deform the day 0 NPV volume
    day0_files = sorted(glob.glob(f'/scratch/rabbit_data/{rabbit}/rawVolumes/Ablation*/*'))
    # day3_files = sorted(glob.glob(f'{in_path}/mri/invivo/volumes/raw/*'))

    # hist_file = f'{in_path}/microscopic/recons/all_ablation_segs_to_invivo.nrrd'
    # if not os.path.exists(hist_file):
    #     hist_file = f'{in_path}/microscopic/recons/all_ablation_segs_to_invivo.mhd'
    interday_file = sorted(glob.glob(f'{in_path}/mri/day0/volumes/raw/*'))
    log_ctd = '/hdscratch/ucair/AcuteBiomarker/Data/18_062/18_062_log_ctd_map.nii.gz'
    t1_nc_file = '/home/sci/blakez/ucair/longitudinal/18_062/Day0_non_contrast_VIBE.nii.gz'

    day0_npv_file = [x for x in day0_files if 'npv' in x or 'NPV' in x and '.nrrd' in x][0]
    day0_t1_file = [x for x in day0_files if '0.5x' in x][-1]
    # day3_npv_file = [x for x in day3_files if 'npv' in x or 'NPV' in x and '.nrrd' in x][0]
    interday = [x for x in interday_file if 'phi' in x and '.mhd' in x][0]

    day0_npv = io.LoadITKFile(day0_npv_file, device)
    day0_t1 = io.LoadITKFile(day0_t1_file, device)
    deform = io.LoadITKFile(interday, device)
    log_ctd = io.LoadITKFile(log_ctd, device)
    day0_nc_t1 = io.LoadITKFile(t1_nc_file, device)

    def_day0 = so.ApplyGrid.Create(deform, device=device)(day0_npv, deform)
    def_day0_t1 = so.ApplyGrid.Create(deform, device=device)(day0_t1, deform)
    def_log_ctd = so.ApplyGrid.Create(deform, device=device)(log_ctd, deform)
    def_nc_t1 = so.ApplyGrid.Create(deform, device=device)(day0_nc_t1, deform)

    to_block = gd.generate(rabbit, block=block, source_space='invivo', target_space='blockface')

    # Get the transformation with the affine included from invivo to exvivo
    # invivo_to_exvivo = mr_to_exvivo(day3_dir)
    # deformations.append(invivo_to_exvivo)

    # Get the transformation with the affine included from exvivo to the stacked blocks
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

    # Load the Day3 MR file to be deformed
    # t2_motion = io.LoadITKFile(
    #     '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/089_----_t2_spc_1mm_iso_cor_post20min.nii.gz',
    #     device=device
    # )
    #
    # ce_t1 = io.LoadITKFile(
    #     '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/102_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_00.nii.gz',
    #     device=device
    # )

    # npv = io.LoadITKFile(
    #     '/hdscratch/ucair/18_062/mri/invivo/volumes/raw/028_----_Day3_lr_NPV_Segmentation_062.nrrd',
    #     device=device
    # )

    # ctd = io.LoadITKFile(
    #     '/home/sci/blakez/ucair/18_047/rawVolumes/Ablation_2018-06-28/CTD_map.nrrd',
    #     device=device
    # )
    # ctd.data[ctd.data < 240] = 0.0
    # ctd.data[ctd.data > 0.0] = 1.0

    torch.cuda.empty_cache()

    # t2_to_block = so.ApplyGrid.Create(invivo_to_block, device=device, pad_mode='zeros')(t2_motion, invivo_to_block)
    # io.SaveITKFile(t2_to_block, f'{invivo_out}day0_t2_to_{block}.mhd')
    #
    # t1_to_block = so.ApplyGrid.Create(invivo_to_block, device=device, pad_mode='zeros')(ce_t1, invivo_to_block)
    # io.SaveITKFile(t1_to_block, f'{invivo_out}day0_ce_t1_to_{block}.mhd')

    npv_to_block = so.ApplyGrid.Create(to_block, device=device, pad_mode='zeros')(def_day0, to_block)
    io.SaveITKFile(npv_to_block, f'{invivo_out}day0_npv_to_{block}.mhd')

    t1_to_block = so.ApplyGrid.Create(to_block, device=device, pad_mode='zeros')(def_day0_t1, to_block)
    io.SaveITKFile(t1_to_block, f'{invivo_out}day0_t1_to_{block}.mhd')

    log_ctd_block = so.ApplyGrid.Create(to_block, device=device, pad_mode='zeros')(def_log_ctd, to_block)
    io.SaveITKFile(log_ctd_block, f'{invivo_out}day0_log_ctd_to_{block}.mhd')

    def_nc_t1 = so.ApplyGrid.Create(to_block, device=device, pad_mode='zeros')(def_nc_t1, to_block)
    io.SaveITKFile(def_nc_t1, f'{invivo_out}day0_t1_nc_to_{block}.mhd')

    # ctd_to_block = so.ApplyGrid.Create(invivo_to_block, device=device, pad_mode='zeros')(ctd, invivo_to_block)
    # io.SaveITKFile(ctd_to_block, f'{invivo_out}day0_ctd_to_{block}.mhd')

    print(f'Done with {block}')


if __name__ == '__main__':
    rabbit = '18_062'
    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    for block_path in [block_list[3]]:
        mr_to_block(block_path, rabbit)
