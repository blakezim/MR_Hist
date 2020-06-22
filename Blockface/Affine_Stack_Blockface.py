import os
import sys
import csv
import ast
sys.path.append("/home/sci/blakez/code/")
import glob
import yaml
import copy
import h5py
import tools
import torch
import shutil
import numpy as np
import subprocess as sp
import skimage.segmentation as seg
from skimage import measure

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


def affine_register(target, source, converge=1.0, niter=300, device='cpu', rigid=True):

    gaussian_blur = so.Gaussian.Create(1, 50, 20, device=device)
    target = gaussian_blur(target)
    source = gaussian_blur(source)

    # Do some additional registration just to make sure it is in the right spot
    similarity = so.L2Similarity.Create(device=device)
    model = so.AffineIntensity.Create(similarity, device=device)

    # Create the optimizer
    optimizer = optim.SGD([
        {'params': model.affine, 'lr': 1.0e-12},
        {'params': model.translation, 'lr': 1.0e-12}], momentum=0.5, nesterov=True
    )

    energy = []
    for epoch in range(0, niter):
        optimizer.zero_grad()
        loss = model(
            target.data, source.data
        )
        energy.append(loss.item())

        print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')

        loss.backward()  # Compute the gradients
        optimizer.step()  #

        if rigid:
            with torch.no_grad():
                U, s, V = model.affine.clone().svd()
                model.affine.data = torch.mm(U, V.transpose(1, 0))

        # if epoch >= 2:
        if epoch > 10 and np.mean(energy[-10:]) - energy[-1] < converge:
            break

    itr_affine = torch.eye(3, device=device, dtype=torch.float32)
    itr_affine[0:2, 0:2] = model.affine
    itr_affine[0:2, 2] = model.translation

    return itr_affine


def solve_affines(block_list):

    for block_path in block_list:
        block = block_path.split('/')[-1]

        if not os.path.exists(f'{block_path}/volumes/raw/'):
            os.makedirs(f'{block_path}/volumes/raw/difference/')
            os.makedirs(f'{block_path}/volumes/raw/surface/')
            os.makedirs(f'{block_path}/volumes/raw/scatter/')

        # elif sorted(glob.glob(f'{block_path}/volumes/raw/difference/*')):
        #     print('alread filtered')
        #     continue

        image_list = sorted(glob.glob(f'{block_path}/images/filtered/*'))

        if not image_list:
            print(f'No filtered image files found for {block} ... skipping')
            continue

        print(f'Solving Affines for {block} ... ')
        spacing = list(map(float, input(f'X,Y Spacing for {block}: ').strip().split(' ')))
        # spacing = [0.0163, 0.0163]

        surface_list = [x for x in image_list if 'surface' in x]
        scatter_list = [x for x in image_list if 'scatter' in x]

        ImScatter = io.LoadITKFile(scatter_list[0], device=device)
        ImScatter.set_spacing_(spacing)
        ImScatter.set_origin_(-1 * (ImScatter.size * ImScatter.spacing) / 2)
        ImScatter /= 255.0

        # Load the surface image
        ImSurface = io.LoadITKFile(surface_list[0], device=device)
        ImSurface.set_spacing_(spacing)
        ImSurface.set_origin_(-1 * (ImSurface.size * ImSurface.spacing) / 2)
        ImSurface /= 255.0

        # Save out the first image
        difference = ImScatter - ImSurface

        ImDifference = core.StructuredGrid(
            ImSurface.shape()[1:],
            tensor=difference.data[2].unsqueeze(0),
            spacing=ImSurface.spacing,
            origin=ImSurface.origin,
            device=device,
            dtype=torch.float32,
            channels=1
        )

        io.SaveITKFile(ImScatter, f'{block_path}/volumes/raw/scatter/IMG_001_scatter.mhd')
        io.SaveITKFile(ImSurface, f'{block_path}/volumes/raw/surface/IMG_001_surface.mhd')
        io.SaveITKFile(ImDifference, f'{block_path}/volumes/raw/difference/IMG_001_difference.mhd')

        ImPrev = ImScatter.copy()

        Adict = {'origin': ImPrev.origin.tolist(), 'spacing': ImPrev.spacing.tolist(),
                 scatter_list.index(scatter_list[0]): np.eye(3).tolist()}

        maxIter = 1000

        for scat, surf in zip(scatter_list[1:], surface_list[1:]):
            print(f'Registering {scat.split("/")[-1]} .... ')
            sys.stdout.flush()
            image_num = scat.split('/')[-1].split('_')[1]

            # Get the number the file is from the start
            dist = scatter_list.index(scat)

            # Load the next image
            ImScatter = io.LoadITKFile(scat, device=device)
            ImScatter.set_spacing_(ImPrev.spacing)
            ImScatter.set_origin_(ImPrev.origin)
            ImScatter /= 255.0

            ImSurface = io.LoadITKFile(surf, device=device)
            ImSurface.set_spacing_(ImPrev.spacing)
            ImSurface.set_origin_(ImPrev.origin)
            ImSurface /= 255.0

            difference = ImScatter - ImSurface

            ImDifference = core.StructuredGrid(
                ImSurface.shape()[1:],
                tensor=difference.data[2].unsqueeze(0),
                spacing=ImSurface.spacing,
                origin=ImSurface.origin,
                device=device,
                dtype=torch.float32,
                channels=1
            )

            affine = affine_register(ImPrev.copy(), ImSurface.copy(), niter=maxIter, device=device)

            # Save out the images
            aff_filter = so.AffineTransform.Create(affine=affine, device=device)
            aff_scatter = aff_filter(ImScatter)
            aff_surface = aff_filter(ImSurface)
            aff_difference = aff_filter(ImDifference)
            # difference = (difference - difference.min()) / (difference.max() - difference.min())

            io.SaveITKFile(aff_scatter, f'{block_path}/volumes/raw/scatter/IMG_{image_num}_scatter.mhd')
            io.SaveITKFile(aff_surface, f'{block_path}/volumes/raw/surface/IMG_{image_num}_surface.mhd')
            io.SaveITKFile(aff_difference, f'{block_path}/volumes/raw/difference/IMG_{image_num}_difference.mhd')

            Adict[dist] = affine.detach().cpu().clone().tolist()
            ImPrev = aff_scatter.copy()

        with open(f'{block_path}/volumes/raw/{block}_affine_dictionary.yaml', 'w') as f:
            yaml.dump(Adict, f)

        _make_volumes(block_path)


def _make_volumes(block_path):

    for vol_type in ['difference', 'surface', 'scatter']:
        mhd_list = sorted(glob.glob(f'{block_path}/volumes/raw/{vol_type}/*.mhd'))

        mhd_dict = tools.read_mhd_header(mhd_list[0])

        # Update the transform matrix
        mhd_dict['TransformMatrix'] = ' '.join(map(str, torch.eye(3).flatten().int().numpy()))

        # Update the offset
        offset = list(map(float, mhd_dict['Offset'].split(' ')))
        offset += [0.0]
        mhd_dict['Offset'] = ' '.join(map(str, offset))

        # Update center of roation
        mhd_dict['CenterOfRotation'] = ' '.join(map(str, [0, 0, 0]))

        # Update number of dimensions
        mhd_dict['NDims'] = 3

        # Update the number of chanells
        if vol_type == 'difference':
            mhd_dict['ElementNumberOfChannels'] = 1
        else:
            mhd_dict['ElementNumberOfChannels'] = 3

        # Update the files to be loaded
        prefix = f'{block_path}/volumes/raw/'
        name = mhd_list[0].split('.')[0].split('_')[-1]
        data_files = f'{name}/IMG_%03d_{name}.raw 1 {len(mhd_list)} 1'
        mhd_dict['ElementDataFile'] = data_files

        # Update the dimensions
        dims = list(map(int, mhd_dict['DimSize'].split(' ')))
        dims += [len(mhd_list)]
        mhd_dict['DimSize'] = ' '.join(map(str, dims))

        # Update the spacing
        spacing = list(map(float, mhd_dict['ElementSpacing'].split(' ')))
        spacing += [0.05]
        mhd_dict['ElementSpacing'] = ' '.join(map(str, spacing))

        tools.write_mhd_header(f'{prefix}{name}_volume.mhd', mhd_dict)


def filter_images(block_list):
    raw_ext = '/images/raw/'
    vol_ext = '/volumes/raw/'
    img_ext = '/images/processed/'

    for block_path in block_list:
        block = block_path.split('/')[-1]

        if not os.path.exists(f'{block_path}/images/filtered/'):
            os.makedirs(f'{block_path}/images/filtered/')

        elif sorted(glob.glob(f'{block_path}/images/filtered/*')):
            print('alread filtered')
            continue

        csv_files = sorted(glob.glob(f'{block_path}/images/csv_files/*'))
        if csv_files == []:
            print(f'No CSV files found for {block} ... skipping')
            continue

        print(f'Filtering {block} ... ', end='')

        # Load the csv file
        image_nums = []
        image_depth = []
        csv_file = csv_files[-1]

        with open(csv_file, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                image_nums.append(int(ast.literal_eval(row[0])[0].split('_')[1]))
                image_depth.append(int(row[1]))

        if len(csv_files) != len(image_nums):
            print(f'The number of CSV files {len(csv_files)} != number of image files {len(image_nums)} ... ')

        filtered_nums = []
        for im in range(0, len(image_nums)):

            if im != len(image_nums) - 1:
                if (image_depth[im] - image_depth[im + 1]) == 0:
                    continue
                else:
                    filtered_nums.append(image_nums[im])

        for im in range(1, len(filtered_nums) + 1):
            new_scatter_file = f'{block_path}/images/filtered/IMG_{im:03d}_scatter.tif'
            new_surface_file = f'{block_path}/images/filtered/IMG_{im:03d}_surface.tif'

            old_scatter_file = f'{block_path}/images/processed/IMG_{filtered_nums[im-1]:04d}_scatter.tif'
            old_surface_file = f'{block_path}/images/processed/IMG_{filtered_nums[im-1]:04d}_surface.tif'

            try:
                shutil.copy(old_scatter_file, new_scatter_file)
            except FileNotFoundError:
                old_scatter_file = f"{old_scatter_file.split('.')[0]}.TIF"
                shutil.copy(old_scatter_file, new_scatter_file)

            try:
                shutil.copy(old_surface_file, new_surface_file)
            except FileNotFoundError:
                old_surface_file = f"{old_surface_file.split('.')[0]}.TIF"
                shutil.copy(old_surface_file, new_surface_file)
        print('done')


if __name__ == '__main__':
    rabbit = '18_062'
    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'


    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))[1:]
    # filter_images(block_list)
    solve_affines(block_list)
