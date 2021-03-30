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
from scipy.stats import ttest_rel
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
        # np_coords[:, 1] *= -1
        # np_coords[:, 0] *= -1
        return torch.as_tensor(np_coords[:, ::-1].copy())


def write_fcsv(data, file):
    # Assume that we need to flip the data and what not
    data = np.array(data.tolist())[:, ::-1]

    # Now need to flip the dimensions to be consistent with slicer
    # data[:, 0] *= -1
    # data[:, 1] *= -1

    with open(file, 'w') as f:
        w = csv.writer(f, delimiter=',')
        w.writerow(['# Markups fiducial file version = 4.10'])
        w.writerow(['# CoordinateSystem = 0'])
        w.writerow(['# columns = id', 'x', 'y', 'z', 'ow', 'ox', 'oy', 'oz', 'vis', 'sel', 'lock', 'label', 'desc',
                    'associatedNodeID'])

        for i, point in enumerate(data):
            prefix = [f'vtkMRMLMarkupsFiducialNode_{i}']
            post = ['0.000', '0.000', '0.000', '1.000', '1', '1', '0', f'F-{i + 1}', '', 'vtkMRMLScalarVolumeNode1']
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


def eval_R3(rabbit_list):
    def_diff = []
    aff_diff = []
    print('Eval R3:')

    for r in rabbit_list:
        print(f'Processing Rabbit {r} ... ', end='')
        in_path = f'/hdscratch/ucair/{r}/mri/'
        if r == '18_047':
            in_path = f'/hdscratch2/{r}/mri/'

        in_landmarks_file = f'{in_path}invivo/landmarks/R3_{r}_invivo_landmarks.fcsv'
        ex_landmarks_file = f'{in_path}exvivo/landmarks/R3_{r}_exvivo_landmarks.fcsv'

        in_landmarks = load_fscv(in_landmarks_file)
        ex_landmarks = load_fscv(ex_landmarks_file)

        if r == '18_047':
            ex_to_in = generate(r, source_space='invivo', target_space='exvivo', base_dir='/hdscratch2/')
            ex_to_in_aff = generate_affine_only(r, source_space='invivo', target_space='exvivo',
                                                base_dir='/hdscratch2/')
        else:
            ex_to_in = generate(r, source_space='invivo', target_space='exvivo')
            ex_to_in_aff = generate_affine_only(r, source_space='invivo', target_space='exvivo')

        def_landmarks = sample_points(ex_to_in, ex_landmarks)
        aff_landmarks = sample_points(ex_to_in_aff, ex_landmarks)

        def_diff.append(def_landmarks.cpu() - in_landmarks.cpu())
        aff_diff.append(aff_landmarks.cpu() - in_landmarks.cpu())

        print('done')

    def_dist = torch.sqrt((torch.cat(def_diff, 0) ** 2).sum(-1))
    aff_dist = torch.sqrt((torch.cat(aff_diff, 0) ** 2).sum(-1))

    print('Eval R3: Done')

    return def_dist, aff_dist


def eval_R2():
    def_diff = []
    aff_diff = []
    rabbit_list = ['18_061']

    print('Eval R2:')
    for r in rabbit_list:

        print(f'Processing Rabbit {r} ... ', end='')
        in_path = f'/hdscratch/ucair/{r}/'
        if r == '18_047':
            in_path = f'/hdscratch2/{r}/'

        block_list = sorted(glob.glob(f'{in_path}/blockface/block*'))

        for block_path in block_list:

            b = block_path.split('/')[-1]

            if not os.path.exists(f'{in_path}blockface/{b}/landmarks/R2_{r}_{b}_blockface_landmarks.fcsv'):
                continue

            bf_landmarks_file = f'{in_path}blockface/{b}/landmarks/R2_{r}_{b}_blockface_landmarks.fcsv'
            ex_landmarks_file = f'{in_path}mri/exvivo/landmarks/R2_{r}_{b}_exvivo_landmarks.fcsv'

            bf_landmarks = load_fscv(bf_landmarks_file)
            ex_landmarks = load_fscv(ex_landmarks_file)

            if r == '18_047':
                bf_to_ex = generate(r, block=b, source_space='exvivo', target_space='blockface',
                                    base_dir='/hdscratch2/')
            else:
                bf_to_ex = generate(r, block=b, source_space='exvivo', target_space='blockface')

            def_landmarks = sample_points(bf_to_ex, bf_landmarks)
            del bf_to_ex
            torch.cuda.empty_cache()

            if r == '18_047':
                bf_to_ex_aff = generate_affine_only(r, block=b, source_space='exvivo', target_space='blockface',
                                                    base_dir='/hdscratch2/')
            else:
                bf_to_ex_aff = generate_affine_only(r, block=b, source_space='exvivo', target_space='blockface')

            aff_landmarks = sample_points(bf_to_ex_aff, bf_landmarks)
            del bf_to_ex_aff
            torch.cuda.empty_cache()

            def_diff.append(def_landmarks.cpu() - ex_landmarks.cpu())
            aff_diff.append(aff_landmarks.cpu() - ex_landmarks.cpu())
        print('done')

    def_dist = torch.sqrt((torch.cat(def_diff, 0) ** 2).sum(-1))
    aff_dist = torch.sqrt((torch.cat(aff_diff, 0) ** 2).sum(-1))

    print('Eval R2: Done')
    return def_dist, aff_dist


def eval_R1(rabbit_list):
    def_diff = []
    aff_diff = []
    print('Eval R1:')

    for r in rabbit_list:
        print(f'Processing Rabbit {r} ... ', end='')
        in_path = f'/hdscratch/ucair/{r}/'
        if r == '18_047':
            in_path = f'/hdscratch2/{r}/'
        block_list = sorted(glob.glob(f'{in_path}/blockface/block*'))
        for b_path in block_list:
            b = b_path.split('/')[-1]
            lm_list = sorted(glob.glob(f'{in_path}/microscopic/{b}/landmarks/R1*'))
            if not lm_list:
                continue
            for lm in lm_list:
                i = lm.split('_')[6]

                r_img = io.LoadITKFile(f'{in_path}microscopic/{b}/segmentations/IMG_{i}/img_{i}_red.nii.gz',
                                       device=device)

                def_dir = f'{in_path}microscopic/{b}/deformations/'
                # Load the affine
                aff = np.loadtxt(glob.glob(f'{def_dir}img_{i}_affine_to_blockface.txt')[0])
                aff = torch.tensor(aff, device=device, dtype=torch.float32)

                # Load the deformation
                deformation_data = io.LoadITKFile(f'{def_dir}/img_{i}_deformation_to_blockface.mhd',
                                                  device=device)

                deformation = core.StructuredGrid(
                    size=deformation_data.size[0:2],
                    spacing=deformation_data.spacing[1:3],
                    origin=deformation_data.origin[1:3],
                    device=deformation_data.device,
                    tensor=deformation_data.data.squeeze().permute(2, 0, 1),
                    channels=2
                )

                # Apply the inverse affine to the grid
                aff = aff.inverse()
                a = aff[0:2, 0:2].float()
                t = aff[-0:2, 2].float()

                aff_deformation = deformation.clone()
                aff_deformation.set_to_identity_lut_()

                deformation.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                                deformation.data.permute(list(range(1, 2 + 1)) + [0]).unsqueeze(-1))
                deformation.data = (deformation.data.squeeze() + t).permute([-1] + list(range(0, 2)))

                aff_deformation.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                                    aff_deformation.data.permute(list(range(1, 2 + 1)) + [0]).unsqueeze(
                                                        -1))
                aff_deformation.data = (aff_deformation.data.squeeze() + t).permute([-1] + list(range(0, 2)))

                in_bf = f'{in_path}blockface/{b}/landmarks/'
                bf_lm = [torch.tensor(np.loadtxt(f'{in_bf}/R1_{r}_{b}_IMG_{i}_blockface_landmarks.txt'), device=device)]

                in_mic = f'{in_path}microscopic/{b}/landmarks/'
                mic_lm = torch.tensor(np.loadtxt(f'{in_mic}/R1_{r}_{b}_IMG_{i}_microscopic_landmarks.txt'),
                                      device=device)

                def_point = sample_points(deformation, bf_lm)
                aff_point = sample_points(aff_deformation, bf_lm)
                # idx_def = def_point[0]/ r_img.spacing

                aff_diff.append((mic_lm - aff_point) * 0.00176)
                def_diff.append((mic_lm - def_point) * 0.00176)  # spacing from histology in mm
        print('done')

    def_dist = torch.sqrt((torch.cat(def_diff, 0) ** 2).sum(-1))
    aff_dist = torch.sqrt((torch.cat(aff_diff, 0) ** 2).sum(-1))

    print('Eval R1: Done')

    return def_dist, aff_dist


def compile_landmarks():
    rabbit_list = ['18_047', '18_060', '18_061', '18_062']

    r3_def, r3_aff = eval_R3(rabbit_list)
    r2_def, r2_aff = eval_R2(rabbit_list)
    r1_def, r1_aff = eval_R1(rabbit_list)

    r1_def = r1_def.reshape(4, 5)
    r1_aff = r1_aff.reshape(4, 5)
    r2_def = r2_def.reshape(4, 5)
    r2_aff = r2_aff.reshape(4, 5)
    r3_def = r3_def.reshape(4, 5)
    r3_aff = r3_aff.reshape(4, 5)

    output_array = [torch.tensor(range(1, len(rabbit_list) + 1)).double()]

    output_array.append(r1_def.cpu().mean(1))
    output_array.append(r1_def.cpu().std(1))
    output_array.append(r1_aff.cpu().mean(1))
    output_array.append(r1_aff.cpu().std(1))
    output_array.append(r2_def.cpu().mean(1))
    output_array.append(r2_def.cpu().std(1))
    output_array.append(r2_aff.cpu().mean(1))
    output_array.append(r2_aff.cpu().std(1))
    output_array.append(r3_def.cpu().mean(1))
    output_array.append(r3_def.cpu().std(1))
    output_array.append(r3_aff.cpu().mean(1))
    output_array.append(r3_aff.cpu().std(1))
    output_array.append(r1_def.cpu().mean(1) + r2_def.cpu().mean(1) + r3_def.cpu().mean(1))
    output_array.append(((r1_def.cpu().var(1) + r2_def.cpu().var(1) + r3_def.cpu().var(1)) / 3).sqrt())
    output_array.append(r1_aff.cpu().mean(1) + r2_aff.cpu().mean(1) + r3_aff.cpu().mean(1))
    output_array.append(((r1_aff.cpu().var(1) + r2_aff.cpu().var(1) + r3_aff.cpu().var(1)) / 3).sqrt())

    output_array = torch.stack(output_array, 1).float()

    hdr = 'num, r1dm, r1ds, r1am, r1as, r2dm, r2ds, r2am, r2as, r3dm, r3ds, r3am, r3as, tds, tdm, tam, tas'

    out = '/home/sci/blakez/papers/histopathologyMethods/Figures/landmark_comp/'
    if not os.path.exists(out):
        os.makedirs(out)

    np.savetxt(f'{out}stats.csv', output_array, delimiter=',', header=hdr, comments='')


def deform_points(base_dir='/hdscratch/ucair/'):
    rabbit_list = ['18_047', '18_060', '18_061', '18_062']
    # bf_dir = f'{base_dir}{rabbit}/blockface/'
    # mr_dir = f'{base_dir}{rabbit}/mri/invivo/landmarks/'
    old_dir = '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/'
    # block_list = sorted(glob.glob(f'{bf_dir}block*'))

    ff_aff_diff = []
    # Affine Landmarks
    for r in rabbit_list:
        in_path = f'/hdscratch/ucair/{r}/'
        if r == '18_047':
            in_path = f'/hdscratch2/{r}/'
        block_list = sorted(glob.glob(f'{in_path}blockface/block*'))
        for block_path in block_list:

            b = block_path.split('/')[-1]

            if not os.path.exists(f'{in_path}blockface/{b}/landmarks/R2_{r}_{b}_blockface_landmarks.fcsv'):
                continue

            bf_landmarks_file = f'{in_path}blockface/{b}/landmarks/R2_{r}_{b}_blockface_landmarks.fcsv'
            ex_landmarks_file = f'{in_path}mri/exvivo/landmarks/R2_{r}_{b}_exvivo_landmarks.fcsv'

            print(f'Processing affine {b} ... ', end='')

            # Load the landmarks
            bf_landmarks = load_fscv(bf_landmarks_file)
            mr_landmarks = load_fscv(ex_landmarks_file)

            # Load the front face affine for stacking the blocks
            ff_stack_aff = np.loadtxt(f'{in_path}blockface/{b}/surfaces/raw/{b}_front_face_tform.txt')
            ff_stack_aff = torch.tensor(ff_stack_aff, device=device, dtype=torch.float32)

            ff_exvivo_aff = np.loadtxt(f'{in_path}blockface/recons/surfaces/raw/blocks_to_exvivo_affine_front_face.txt')
            ff_exvivo_aff = torch.tensor(ff_exvivo_aff, device=device, dtype=torch.float32)

            final_aff = ff_stack_aff.clone().cpu()
            final_aff = torch.matmul(ff_exvivo_aff.cpu(), final_aff)

            aff_lm = []
            for lm in bf_landmarks:
                flip_lm = torch.cat([lm.flip(0).float().cpu(), torch.tensor([1.0])]).cpu().unsqueeze(-1)
                tform_lm = torch.matmul(final_aff.cpu(), flip_lm)
                aff_lm.append(tform_lm[:-1].flip(0).permute(1, 0))

            aff_lm = torch.cat(aff_lm, 0)

            diff = aff_lm.cpu() - mr_landmarks.cpu()
            ff_aff_diff.append(diff)
            print('done')

    print('done')

    def_dist, aff_dist = eval_R2(rabbit_list)
    ff_aff_dist = torch.sqrt((torch.cat(ff_aff_diff, 0) ** 2).sum(-1))
    ff_z_dist =(torch.cat(ff_aff_diff, 0))[:, 0].abs()

    def_dist = def_dist.reshape(4, 5)
    ff_aff_dist = ff_aff_dist.reshape(4, 5)
    ff_z_dist = ff_z_dist.reshape(4, 5)

    output_array = [torch.tensor(range(1, len(rabbit_list) + 1)).double()]

    output_array.append(def_dist.cpu().mean(1))
    output_array.append(def_dist.cpu().std(1))
    output_array.append(ff_aff_dist.cpu().mean(1))
    output_array.append(ff_aff_dist.cpu().std(1))
    output_array.append(ff_z_dist.cpu().mean(1))
    output_array.append(ff_z_dist.cpu().std(1))
    output_array.append(ff_aff_dist.cpu().mean(1) - ff_z_dist.cpu().mean(1))
    # output_array.append(r2_aff.cpu().mean(1))
    # output_array.append(r2_aff.cpu().std(1))
    # output_array.append(r3_def.cpu().mean(1))
    # output_array.append(r3_def.cpu().std(1))
    # output_array.append(r3_aff.cpu().mean(1))
    # output_array.append(r3_aff.cpu().std(1))
    # output_array.append(r1_def.cpu().mean(1) + r2_def.cpu().mean(1) + r3_def.cpu().mean(1))
    # output_array.append(((r1_def.cpu().var(1) + r2_def.cpu().var(1) + r3_def.cpu().var(1)) / 3).sqrt())
    # output_array.append(r1_aff.cpu().mean(1) + r2_aff.cpu().mean(1) + r3_aff.cpu().mean(1))
    # output_array.append(((r1_aff.cpu().var(1) + r2_aff.cpu().var(1) + r3_aff.cpu().var(1)) / 3).sqrt())

    output_array = torch.stack(output_array, 1).float()

    hdr = 'num, r2dm, r2ds, fftam, fftas, ffzam, ffzas, ffsubs'

    out = '/home/sci/blakez/papers/histopathologyMethods/Figures/projection/'
    if not os.path.exists(out):
        os.makedirs(out)

    np.savetxt(f'{out}stats.csv', output_array, delimiter=',', header=hdr, comments='')

    # df = pd.DataFrame(pd_dict)
    # df['lm_num'] = [f'lm{x:02d}' for x in range(1, len(def_dist) + 1)] * 4
    # results = AnovaRM(df, depvar='Dep_Var', subject='lm_num', within=['Def_ID'])
    # print(results.fit())

    # stacked_dists = torch.stack((def_dist, aff_dist, ff_def_dist, ff_aff_dist), dim=-1)
    # df = pd.DataFrame(data=stacked_dists.numpy(), index=range(0, 10), columns=['def', 'aff', 'ff_def', 'ff_aff'])
    # df['id'] = [f'lm{x:02d}' for x in range(1, len(def_dist) + 1)]
    # df = pd.DataFrame(data=cat_dists.numpy(), index=range(0, 40), columns=['depvar'])
    # df['id'] = indices
    # df['cond'] = ['noise'] * 40


if __name__ == '__main__':
    # compile_landmarks()
    # deform_points(base_dir='/hdscratch2/')
    # eval_R3()
    eval_R2()
    # eval_R1()
    # generate_r1()
