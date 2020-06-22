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


def corrections(rabbit_dir, block, corr_num):

    block_path = f'{rabbit_dir}{block}/'

    image_list = sorted(glob.glob(f'{block_path}/images/filtered/*'))

    with open(f'{block_path}/volumes/raw/{block}_affine_dictionary.yaml', 'r') as f:
        Adict = yaml.load(f, Loader=yaml.FullLoader)

    surface_list = [x for x in image_list if 'surface' in x]
    scatter_list = [x for x in image_list if 'scatter' in x]

    # Load the surface image
    ImSurface_good = io.LoadITKFile(surface_list[corr_num - 1], device=device)
    ImSurface_good.set_spacing_(Adict['spacing'])
    ImSurface_good.set_origin_(-1 * (ImSurface_good.size * ImSurface_good.spacing) / 2)
    ImSurface_good /= 255.0

    ImSurface_bad = io.LoadITKFile(surface_list[corr_num], device=device)
    ImSurface_bad.set_spacing_(Adict['spacing'])
    ImSurface_bad.set_origin_(-1 * (ImSurface_bad.size * ImSurface_bad.spacing) / 2)
    ImSurface_bad /= 255.0

    # Apply the affines to both of the images
    aff_filter = so.AffineTransform.Create(affine=torch.tensor(Adict[corr_num - 1], device=device), device=device)
    ImSurface_good = aff_filter(ImSurface_good)

    aff_filter = so.AffineTransform.Create(affine=torch.tensor(Adict[corr_num], device=device), device=device)
    ImSurface_bad = aff_filter(ImSurface_bad)

    points = torch.tensor(
        tools.LandmarkPicker([ImSurface_bad[0].squeeze().cpu(), ImSurface_good[0].squeeze().cpu()]),
        dtype=torch.float32,
        device=device
    ).permute(1, 0, 2)

    # Change to real coordinates
    points *= torch.cat([ImSurface_good.spacing[None, None, :], ImSurface_good.spacing[None, None, :]], 0)
    points += torch.cat([ImSurface_good.origin[None, None, :], ImSurface_good.origin[None, None, :]], 0)

    aff_filter = so.AffineTransform.Create(points[1], points[0], device=device)

    corr_affine = torch.eye(3, device=device, dtype=torch.float32)
    corr_affine[0:2, 0:2] = aff_filter.affine
    corr_affine[0:2, 2] = aff_filter.translation

    ImPrev = ImSurface_good.copy()

    for scat, surf in zip(scatter_list[corr_num:], surface_list[corr_num:]):
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

        affine = torch.mm(corr_affine, torch.tensor(Adict[dist], device=device))

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


if __name__ == '__main__':
    rabbit = '18_062'
    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    block = 'block06'
    corr_num = 2
    corrections(rabbit_dir, block, corr_num=corr_num)
