import os
import csv
import sys
sys.path.append("/home/sci/blakez/code/")
import glob
import yaml
import copy
import h5py
import tools
import torch
import numpy as np
import torch.nn.functional as F
import subprocess as sp
import skimage.segmentation as seg

from collections import OrderedDict
import torch.optim as optim

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridTools as st
import CAMP.UnstructuredGridOperators as uo
import CAMP.StructuredGridOperators as so

# import matplotlib
# matplotlib.use('qt5agg')
# import matplotlib.pyplot as plt
# plt.ion()

device = 'cuda:1'

#### ADDED ####
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


def get_deformation(block_path, rabbit):

    day3_dir = f'/hdscratch/ucair/{rabbit}/mri/invivo/'
    stacked_blocks_dir = f'/hdscratch/ucair/{rabbit}/blockface/recons/'
    block = block_path.split('/')[-1]
    exvivo_out = f'/hdscratch/ucair/18_047/mri/exvivo/volumes/deformable/{block}/'
    invivo_out = f'/hdscratch/ucair/18_047/mri/invivo/volumes/deformable/{block}/'

    deformations = []

    # Get the transformation with the affine included from invivo to exvivo
    invivo_to_exvivo = mr_to_exvivo(day3_dir)
    deformations.append(invivo_to_exvivo)

    # Get the transformation with the affine included from exvivo to the stacked blocks
    exvivo_to_stacked = exvivo_to_blocks(stacked_blocks_dir)
    deformations.append(exvivo_to_stacked)

    # Load the transformation from the stacked blocks to the block of interest
    stacked_to_block = stacked_blocks_to_block(block_path)
    deformations.append(stacked_to_block)

    invivo_to_block = so.ComposeGrids.Create(device=device)(deformations[::-1])

    return invivo_to_block

def load_fscv(file):
    with open(file, 'r') as f:
        lines = csv.reader(f, delimiter=',')
        coord = []
        for i, row in enumerate(lines):
            if i < 3:
                pass
            elif any(np.isnan(np.array([float(x) for x in row[1:4]]))):
                pass
            else:
                coord.append([float(x) for x in row[1:4]])

        np_coords = np.array(coord)
        np_coords[:, 1] *= -1
        np_coords[:, 0] *= -1
        return torch.as_tensor(np_coords[:, ::-1].copy())


def write_fcsv(data, file):
    # Assume that we need to flip the data and what not
    data = np.array(data.tolist())[:, ::-1]

    # Now need to flip the dimensions to be consistent with slicer
    data[:, 0] *= -1
    data[:, 1] *= -1

    with open(file, 'w') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(['# Markups fiducial file version = 4.10'])
        w.writerow(['# CoordinateSystem = 0'])
        w.writerow(['# columns = id', 'x', 'y', 'z', 'ow', 'ox', 'oy', 'oz', 'vis', 'sel','lock', 'label', 'desc', 'associatedNodeID'])

        for i, point in enumerate(data):
            prefix = [f'vtkMRMLMarkupsFiducialNode_{i}']
            post = ['0.000', '0.000', '0.000', '1.000', '1', '1', '0', f'F-{i+1}', '', 'vtkMRMLScalarVolumeNode1']
            point = [str(x) for x in point]
            w.writerow(prefix + point + post)


def sample_points(field, points):
    out_points = []
    for point in points:
        point = torch.as_tensor(point)
        point = point.to(field.device)
        point = point.type(field.dtype)

        # Change to index coordinate
        index_point = ((point - field.origin) / field.spacing) - 1
        torch_point = (index_point / (field.size / 2) - 1)

        # We turn it into a torch coordinate, but now grid sample is expecting x, y, z

        torch_point = torch.as_tensor((torch_point.tolist()[::-1]), device=field.device, dtype=field.dtype)
        out_point = F.grid_sample(field.data.unsqueeze(0),
                                  torch_point.view([1] * (len(field.size) + 1) + [len(field.size)]),
                                  align_corners=True)

        out_points.append(out_point.squeeze())

    out_points = torch.stack(out_points, 0)
    return out_points


def deform_points(rabbit):

    bf_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    mr_dir = f'/hdscratch/ucair/{rabbit}/mri/invivo/landmarks/'
    old_dir = '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/'
    block_list = sorted(glob.glob(f'{bf_dir}block*'))

    for block_path in block_list:
        block = block_path.split('/')[-1]
        # Check to see if there are any landmarks for this block
        bf_landmark_file = sorted(glob.glob(f'{bf_dir}{block}/landmarks/*.fcsv'))
        if bf_landmark_file == []:
            print(f'No landmarks for {block}')
            continue

        # Load the landmarks
        bf_landmarks = load_fscv(bf_landmark_file[0]).to(device)
        mr_landmark_file = sorted(glob.glob(f'{mr_dir}{block}*.fcsv'))
        mr_landmarks = load_fscv(mr_landmark_file[0])

        # Get the deformation from the block to the invivo
        # rabbit = '18_047'
        # rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'

        # Get a list of the blocks
        from_mr_to_block = get_deformation(block_path, rabbit)

        hope = sample_points(from_mr_to_block, bf_landmarks)

        diff = hope.cpu() - mr_landmarks.cpu()
        print(diff)
        print(torch.sqrt((diff ** 2).sum(-1)))

        # # Load the affine that goes from stacked to exvivo
        # aff_to_ex = np.loadtxt(f'{old_dir}blocks_to_exvivo_affine.txt')
        # aff_to_ex = torch.tensor(aff_to_ex, device=device, dtype=torch.float32)
        #
        # phi_inv_ex = io.LoadITKFile(f'{old_dir}block_phi_inv.mhd', device=device)
        # phi_inv_ex.data = phi_inv_ex.data.flip(0)
        #
        # aff_to_ex = aff_to_ex.inverse()
        # a = aff_to_ex[0:3, 0:3].float()
        # t = aff_to_ex[-0:3, 3].float()
        #
        # phi_inv_ex.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
        #                                phi_inv_ex.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        # phi_inv_ex.data = (phi_inv_ex.data.squeeze() + t).permute([-1] + list(range(0, 3)))
        # phi_inv_ex.data = phi_inv_ex.data.flip(0)
        #
        # to_block_mr_ponts = sample_points(phi_inv_ex, mr_landmarks)
        #
        # # Load the affine
        # aff = np.loadtxt(f'{bf_dir}{block}/surfaces/raw/{block}_rigid_tform.txt')
        # aff = torch.tensor(aff, device=device, dtype=torch.float32)
        #
        # # Load phi_inv
        # phi_inv = io.LoadITKFile(f'{bf_dir}{block}/volumes/raw/{block}_phi_inv.mhd', device=device)
        # phi_inv.set_size((60, 1024, 1024))
        # #
        # # # Phi_inv is in z,y,x and need x,y,z
        # phi_inv.data = phi_inv.data.flip(0)
        #
        # # Apply the inverse affine to the grid
        # aff = aff.inverse()
        # a = aff[0:3, 0:3].float()
        # t = aff[-0:3, 3].float()
        #
        # phi_inv.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
        #                             phi_inv.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        # phi_inv.data = (phi_inv.data.squeeze() + t).permute([-1] + list(range(0, 3)))
        #
        # # Flip phi_inv back to the way it was
        # phi_inv.data = phi_inv.data.flip(0)



def exvivo_to_invivo(rabbit):

    data_dir = f'/hdscratch/ucair/{rabbit}/mri/invivo/'

    # Load the landmarks in exvivo
    ex_points_file = sorted(glob.glob(f'{data_dir}landmarks/*exvivo.fcsv'))
    ex_points = load_fscv(ex_points_file[0])

    # Load the affine that goes from exvivo to invivo
    aff = np.loadtxt(f'{data_dir}surfaces/raw/invivo_to_exvivo_affine.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    # Load the transformation
    phi_inv = io.LoadITKFile(f'{data_dir}volumes/deformable/invivo_phi_inv.mhd', device=device)

    # # Phi_inv is in z,y,x and need x,y,z
    phi_inv.data = phi_inv.data.flip(0)

    # Apply the inverse affine to the grid
    aff = aff.inverse()
    a = aff[0:3, 0:3].float()
    t = aff[-0:3, 3].float()

    phi_inv.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                phi_inv.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    phi_inv.data = (phi_inv.data.squeeze() + t).permute([-1] + list(range(0, 3)))

    phi_inv.data = phi_inv.data.flip(0)

    invivo_points = sample_points(phi_inv, ex_points)

    # Write out the points
    write_fcsv(invivo_points, f'{data_dir}landmarks/block08_landmarks_invivo.fcsv')


if __name__ == '__main__':
    rabbit = '18_047'
    deform_points(rabbit)
    # exvivo_to_invivo(rabbit)