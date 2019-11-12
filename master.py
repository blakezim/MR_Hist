import os
import sys
import glob
import copy
import tools
import torch
import numpy as np
import subprocess as sp

from collections import OrderedDict
import torch.optim as optim

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.UnstructuredGridOperators as uo
import CAMP.StructuredGridOperators as so

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()


device = 'cuda:1'

def main(rabbit):

    rabbit_dir = f'/hdscratch/ucair/blockface/{rabbit}'
    raw_dir = f'{rabbit_dir}/surfaces/raw/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{raw_dir}*'))

    # Determine the middle block
    middle_block = block_list[len(block_list) // 2].split('/')[-1]
    head_block = block_list[(len(block_list) // 2) + 1].split('/')[-1]

    # Load the surfaces
    tf_verts, tf_faces = io.ReadOBJ(f'{raw_dir}{middle_block}/{middle_block}_surface_foot.obj')
    sh_verts, sh_faces = io.ReadOBJ(f'{raw_dir}{head_block}/{head_block}_surface_head.obj')

    # Load the other objects to deform as well
    se_verts, se_faces = io.ReadOBJ(f'{raw_dir}{head_block}/{head_block}_surface_exterior.obj')
    sf_verts, sf_faces = io.ReadOBJ(f'{raw_dir}{head_block}/{head_block}_surface_foot.obj')

    # Create surface objects
    tar_surface = core.TriangleMesh(tf_verts, tf_faces)
    src_surface = core.TriangleMesh(sh_verts, sh_faces)

    # Create the exterior surface
    resid_surface = core.TriangleMesh(se_verts, se_faces)
    resid_surface.add_surface_(sf_verts, sf_faces)

    tar_surface.to_(device)
    src_surface.to_(device)
    resid_surface.to_(device)

    # Have to flip the normals of one of the surfaces
    src_surface.flip_normals_()

    sigma = torch.tensor([0.5, 0.5, 5.0], device=device)

    affine, deformation, def_surface, def_resid = tools.register_surfaces(
        tar_element=tar_surface,
        src_element=src_surface,
        sigma=sigma,
        src_excess=resid_surface,
        device=device
    )

if __name__ == '__main__':
    rabbit = '18_047'
    main(rabbit)
