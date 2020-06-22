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
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import torch.nn.functional as F
import subprocess as sp
import skimage.segmentation as seg
from GenerateDeformation import generate, generate_affine_only

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

    def_diff = []
    # Deformable Landmarks
    for block_path in block_list:

        block = block_path.split('/')[-1]

        # Check to see if there are any landmarks for this block
        bf_landmark_file = sorted(glob.glob(f'{bf_dir}{block}/landmarks/*.fcsv'))
        if bf_landmark_file == []:
            print(f'No landmarks for {block}')
            continue

        # if block in ['block07', 'block08', 'block09', 'block10']:
        #     continue

        print(f'Processing deformable {block} ... ', end='')

        # Load the landmarks
        bf_landmarks = load_fscv(bf_landmark_file[0]).to(device)
        mr_landmark_file = sorted(glob.glob(f'{mr_dir}{block}*.fcsv'))
        mr_landmarks = load_fscv(mr_landmark_file[0])

        # Get the deformation from the block to the invivo
        from_mr_to_block = generate(rabbit, block=block, source_space='invivo', target_space='blockface')

        def_landmarks = sample_points(from_mr_to_block, bf_landmarks)

        diff = def_landmarks.cpu() - mr_landmarks.cpu()
        def_diff.append(diff)
        print('done')

    aff_diff = []
    # Affine Landmarks
    for block_path in block_list:

        block = block_path.split('/')[-1]

        # Check to see if there are any landmarks for this block
        bf_landmark_file = sorted(glob.glob(f'{bf_dir}{block}/landmarks/*.fcsv'))
        if bf_landmark_file == []:
            print(f'No landmarks for {block}')
            continue

        print(f'Processing affine {block} ... ', end='')

        # Load the landmarks
        bf_landmarks = load_fscv(bf_landmark_file[0]).to(device)
        mr_landmark_file = sorted(glob.glob(f'{mr_dir}{block}*.fcsv'))
        mr_landmarks = load_fscv(mr_landmark_file[0])

        block_to_mr_affine = generate_affine_only(rabbit, block=block, source_space='blockface', target_space='invivo')

        aff_lm = []
        for lm in bf_landmarks:
            flip_lm = torch.cat([lm.flip(0).float().cpu(), torch.tensor([1.0])]).cpu().unsqueeze(-1)
            tform_lm = torch.matmul(block_to_mr_affine.cpu(), flip_lm)
            aff_lm.append(tform_lm[:-1].flip(0).permute(1, 0))

        aff_lm = torch.cat(aff_lm, 0)

        diff = aff_lm.cpu() - mr_landmarks.cpu()
        aff_diff.append(diff)

        print('done')

    ff_aff_diff = []
    # Affine Landmarks
    for block_path in block_list:

        block = block_path.split('/')[-1]

        # Check to see if there are any landmarks for this block
        bf_landmark_file = sorted(glob.glob(f'{bf_dir}{block}/landmarks/*.fcsv'))
        if bf_landmark_file == []:
            print(f'No landmarks for {block}')
            continue

        print(f'Processing affine {block} ... ', end='')

        # Load the landmarks
        bf_landmarks = load_fscv(bf_landmark_file[0]).to(device)
        mr_landmark_file = sorted(glob.glob(f'{mr_dir}{block}*.fcsv'))
        mr_landmarks = load_fscv(mr_landmark_file[0])

        # Load the front face affine for stacking the blocks
        ff_stack_aff = np.loadtxt(f'{block_path}/surfaces/raw/{block}_front_face_tform.txt')
        ff_stack_aff = torch.tensor(ff_stack_aff, device=device, dtype=torch.float32)

        ff_exvivo_aff = np.loadtxt(f'{block_path}/../recons/blocks_to_exvivo_affine_front_face.txt')
        ff_exvivo_aff = torch.tensor(ff_exvivo_aff, device=device, dtype=torch.float32)

        block_to_mr_affine = generate_affine_only(rabbit, block=block, source_space='exvivo', target_space='invivo')
        block_to_mr_affine = block_to_mr_affine.float()

        final_aff = ff_stack_aff.clone().cpu()
        final_aff = torch.matmul(ff_exvivo_aff.cpu(), final_aff)
        final_aff = torch.matmul(block_to_mr_affine.cpu(), final_aff)

        aff_lm = []
        for lm in bf_landmarks:
            flip_lm = torch.cat([lm.flip(0).float().cpu(), torch.tensor([1.0])]).cpu().unsqueeze(-1)
            tform_lm = torch.matmul(final_aff.cpu(), flip_lm)
            aff_lm.append(tform_lm[:-1].flip(0).permute(1, 0))

        aff_lm = torch.cat(aff_lm, 0)

        diff = aff_lm.cpu() - mr_landmarks.cpu()
        ff_aff_diff.append(diff)

        print('done')

    ff_def_diff = []
    # Affine Landmarks
    for block_path in block_list:

        block = block_path.split('/')[-1]

        # Check to see if there are any landmarks for this block
        bf_landmark_file = sorted(glob.glob(f'{bf_dir}{block}/landmarks/*.fcsv'))
        if bf_landmark_file == []:
            print(f'No landmarks for {block}')
            continue

        print(f'Processing affine {block} ... ', end='')

        # Load the landmarks
        bf_landmarks = load_fscv(bf_landmark_file[0]).to(device)
        mr_landmark_file = sorted(glob.glob(f'{mr_dir}{block}*.fcsv'))
        mr_landmarks = load_fscv(mr_landmark_file[0])

        # Load the front face affine for stacking the blocks
        ff_stack_aff = np.loadtxt(f'{block_path}/surfaces/raw/{block}_front_face_tform.txt')
        ff_stack_aff = torch.tensor(ff_stack_aff, device=device, dtype=torch.float32)

        ff_exvivo_aff = np.loadtxt(f'{block_path}/../recons/blocks_to_exvivo_affine_front_face.txt')
        ff_exvivo_aff = torch.tensor(ff_exvivo_aff, device=device, dtype=torch.float32)

        final_aff = ff_stack_aff.clone().cpu()
        final_aff = torch.matmul(ff_exvivo_aff.cpu(), final_aff)

        aff_lm = []
        for lm in bf_landmarks:
            flip_lm = torch.cat([lm.flip(0).float().cpu(), torch.tensor([1.0])]).cpu().unsqueeze(-1)
            tform_lm = torch.matmul(final_aff.cpu(), flip_lm)
            aff_lm.append(tform_lm[:-1].flip(0).permute(1, 0))

        aff_lm = torch.cat(aff_lm, 0)

        ex_in_aff = np.loadtxt('/hdscratch/ucair/18_047/mri/exvivo/surfaces/raw/exvivo_to_invivo_affine.txt')
        ex_in_aff = torch.tensor(ex_in_aff, device=device, dtype=torch.float32)

        # Apply the FORWARD affine to the deformation
        a = ex_in_aff[0:3, 0:3].float()
        t = ex_in_aff[-0:3, 3].float()

        # Create a deformation from the affine that lives in the stacked blocks space
        aff_grid = io.LoadITKFile('/hdscratch/ucair/18_047/mri/exvivo/volumes/raw/reference_volume.nii.gz',
                                  device=device)
        aff_grid.set_to_identity_lut_()
        aff_grid.data = aff_grid.data.flip(0)
        aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
        aff_grid.data = (aff_grid.data.squeeze() + t).permute([-1] + list(range(0, 3)))
        aff_grid.data = aff_grid.data.flip(0)

        exvivo_to_invivo = io.LoadITKFile('/hdscratch/ucair/18_047/mri/exvivo/volumes/raw/exvivo_to_invivo_phi.mhd',
                                          device=device)
        # Compose the grids
        deformation = so.ComposeGrids.Create(device=device)([aff_grid, exvivo_to_invivo])

        def_aff_lm = sample_points(deformation, aff_lm)

        diff = def_aff_lm.cpu() - mr_landmarks.cpu()
        ff_def_diff.append(diff)

        print('done')

    aff_dist = torch.sqrt((torch.cat(aff_diff, 0) ** 2).sum(-1))
    aff_mean = aff_dist.mean()
    aff_std = aff_dist.std()

    def_dist = torch.sqrt((torch.cat(def_diff, 0) ** 2).sum(-1))
    def_mean = def_dist.mean()
    def_std = def_dist.std()

    ff_aff_dist = torch.sqrt((torch.cat(ff_aff_diff, 0) ** 2).sum(-1))
    ff_aff_mean = ff_aff_dist.mean()
    ff_aff_std = ff_aff_dist.std()

    ff_def_dist = torch.sqrt((torch.cat(ff_def_diff, 0) ** 2).sum(-1))
    ff_def_mean = ff_def_dist.mean()
    ff_def_std = ff_def_dist.std()

    indices = [1] * len(aff_dist) + [2] * len(aff_dist) + [3] * len(aff_dist) + [4] * len(aff_dist)
    cat_dists = torch.cat((def_dist, aff_dist, ff_def_dist, ff_aff_dist), dim=0)

    pd_dict = {
        'Def_ID': indices,
        'Dep_Var': cat_dists.numpy()
    }

    df = pd.DataFrame(pd_dict)
    df['lm_num'] = [f'lm{x:02d}' for x in range(1, len(def_dist) + 1)] * 4
    results = AnovaRM(df, depvar='Dep_Var', subject='lm_num', within=['Def_ID'])
    print(results.fit())

    # stacked_dists = torch.stack((def_dist, aff_dist, ff_def_dist, ff_aff_dist), dim=-1)
    # df = pd.DataFrame(data=stacked_dists.numpy(), index=range(0, 10), columns=['def', 'aff', 'ff_def', 'ff_aff'])
    # df['id'] = [f'lm{x:02d}' for x in range(1, len(def_dist) + 1)]
    # df = pd.DataFrame(data=cat_dists.numpy(), index=range(0, 40), columns=['depvar'])
    # df['id'] = indices
    # df['cond'] = ['noise'] * 40

if __name__ == '__main__':
    rabbit = '18_047'
    deform_points(rabbit)
