import os
import sys
sys.path.append("/home/sci/blakez/code/")
import glob
import yaml
import copy
import h5py
import tools
import torch
import shutil
import numpy as np
from torch.autograd import Variable
from CAMP.UnstructuredGridOperators import *
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


def front_face_register(tar_surface, src_surface, affine_lr=1.0e-06, translation_lr=1.0e-04, converge=0.01,
                        spatial_sigma=[0.5], device='cpu', plot=True):
    # Plot the surfaces
    if plot:
        [_, fig, ax] = io.PlotSurface(tar_surface.vertices, tar_surface.indices)
        [src_mesh, _, _] = io.PlotSurface(src_surface.vertices, src_surface.indices, fig=fig, ax=ax, color=[1, 0, 0])

    # Find the inital translation
    init_translation = (tar_surface.centers.mean(0) - src_surface.centers.mean(0)).clone()

    # Set the inital angle
    init_angle = torch.tensor(0.0, dtype=torch.float32, device=device)

    for sigma in spatial_sigma:

        # Create some of the filters
        model = SingleAngleCurrents.Create(
            tar_surface.normals,
            tar_surface.centers,
            sigma=sigma,
            init_angle=init_angle,
            init_translation=init_translation,
            kernel='cauchy',
            device=device
        )

        # Create the optimizer
        optimizer = optim.SGD([
            {'params': model.angle, 'lr': affine_lr},
            {'params': model.translation, 'lr': translation_lr}], momentum=0.9, nesterov=True
        )

        energy = [model.currents.e3.item()]
        for epoch in range(0, 1000):
            optimizer.zero_grad()

            loss = model(
                src_surface.normals.clone(), src_surface.centers.clone()
            )

            print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')
            energy.append(loss.item())

            loss.backward()  # Compute the gradients
            optimizer.step()  # Apply the gradients

            # if rigid:
            #     with torch.no_grad():
            #         U, s, V = model.affine.clone().svd()
            #         model.affine.data = torch.mm(U, V.transpose(1, 0))

            if epoch > 10 and np.mean(energy[-10:]) - energy[-1] < converge:
                break

        # Update the affine and translation for the next sigma
        init_angle = model.angle.detach().clone()
        init_translation = model.translation.detach().clone()

    # # Need to update the translation to account for not rotation about the origin
    affine = model.affine.detach()
    translation = model.translation.detach()
    translation = -torch.matmul(affine, src_surface.centers.mean(0)) + src_surface.centers.mean(0) + translation

    # Construct a single affine matrix
    full_aff = torch.eye(len(affine) + 1)
    full_aff[0:len(affine), 0:len(affine)] = affine.clone()
    full_aff[0:len(affine), len(affine)] = translation.clone().t()

    # Create affine applier filter and apply
    aff_tfrom = uo.AffineTransformSurface.Create(full_aff, device=device)
    aff_source_head = aff_tfrom(src_surface)
    if plot:
        src_mesh.set_verts(aff_source_head.vertices[aff_source_head.indices].detach().cpu().numpy())

    return full_aff


def front_face_stacking(rabbit):

    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_ext = '/surfaces/raw/'
    # rigid_ext = '/surfaces/rigid/'
    ff_ext = '/surfaces/frontface/'
    deform_ext = '/surfaces/deformable/'
    vol_ext = '/volumes/raw/'
    stitch_ext = '/surfaces/raw/stitching/deformable/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    # Determine the middle block
    middle_block = block_list[9]
    foot_blocks = block_list[block_list.index(middle_block):]
    head_blocks = block_list[:block_list.index(middle_block) + 1][::-1]

    rerun = True

    # skip_blocks = ['block05', 'block06', 'block07', 'block08', 'block09', 'block10', 'block11', 'block12']

    skip_blocks = []

    if rerun:
        mid_block = middle_block.split('/')[-1]

        if not os.path.exists(f'{rabbit_dir}{mid_block}{ff_ext}'):
            os.makedirs(f'{rabbit_dir}{mid_block}{ff_ext}')

        affine_tform = torch.eye(4)
        np.savetxt(f'{rabbit_dir}{mid_block}{raw_ext}{mid_block}_front_face_tform.txt', affine_tform.numpy())

        # Copy the files from raw to deformable for the middle surface
        if os.path.exists(f'{rabbit_dir}{mid_block}{stitch_ext}'):
            mid_path = f'{mid_block}{stitch_ext}'
        else:
            mid_path = f'{mid_block}{raw_ext}'

        files = [
            f'{rabbit_dir}{mid_path}{mid_block}_decimate.obj',
            f'{rabbit_dir}{mid_path}{mid_block}_ext.obj',
        ]
        if os.path.exists(f'{rabbit_dir}{mid_path}{mid_block}_foot.obj'):
            files += [f'{rabbit_dir}{mid_path}{mid_block}_foot.obj']
        if os.path.exists(f'{rabbit_dir}{mid_path}{mid_block}_head.obj'):
            files += [f'{rabbit_dir}{mid_path}{mid_block}_head.obj']
        if os.path.exists(f'{rabbit_dir}{mid_path}{mid_block}_foot_support.obj'):
            files += [f'{rabbit_dir}{mid_path}{mid_block}_foot_support.obj']
        if os.path.exists(f'{rabbit_dir}{mid_path}{mid_block}_head_support.obj'):
            files += [f'{rabbit_dir}{mid_path}{mid_block}_head_support.obj']

        out_names = []
        for path in files:
            name = path.split('/')[-1].split(f'{mid_block}')[-1].replace('.', '_front_face.')
            out_path = f'{rabbit_dir}{mid_block}{ff_ext}{mid_block}'
            out_names += [f'{out_path}{name}']

        for in_file, out_file in zip(files, out_names):
            shutil.copy(in_file, out_file)

    # Loop over the foot blocks
    for i, block_path in enumerate(foot_blocks, 1):

        if i == len(foot_blocks):
            break

        target_block = block_path.split('/')[-1]
        source_block = foot_blocks[i].split('/')[-1]

        if source_block in skip_blocks:
            continue

        if os.path.exists(f'{rabbit_dir}{source_block}{stitch_ext}'):
            mid_path = f'{source_block}{stitch_ext}'
        else:
            mid_path = f'{source_block}{raw_ext}'

        target_surface_path = f'{rabbit_dir}{target_block}{ff_ext}{target_block}_foot_front_face.obj'
        source_surface_path = f'{rabbit_dir}{mid_path}{source_block}_head.obj'

        extras_paths = [
            f'{rabbit_dir}{mid_path}{source_block}_decimate.obj',
            f'{rabbit_dir}{mid_path}{source_block}_ext.obj',
        ]

        if i < len(foot_blocks) - 1:
            extras_paths += [f'{rabbit_dir}{mid_path}{source_block}_foot.obj']

        if os.path.exists(f'{rabbit_dir}{mid_path}{source_block}_foot_support.obj'):
            extras_paths += [f'{rabbit_dir}{mid_path}{source_block}_foot_support.obj']

        if os.path.exists(f'{rabbit_dir}{source_block}{ff_ext}{source_block}_head_front_face.obj') and not rerun:
            print(f'The front face surface for {source_block} already exists ... Next block')
            continue

        try:
            verts, faces = io.ReadOBJ(target_surface_path)
            tar_surface = core.TriangleMesh(verts, faces)
            tar_surface.to_(device)
        except IOError:
            print(f'The front face foot surface for {target_block} was not found ... Next block')
            continue

        # Need to see if the target needs any support
        support_block = block_list[block_list.index(block_path) - 1].split('/')[-1]
        if os.path.exists(f'{rabbit_dir}{support_block}{ff_ext}{support_block}_foot_support_front_face.obj'):
            verts, faces = io.ReadOBJ(
                f'{rabbit_dir}{support_block}{ff_ext}{support_block}_foot_support_front_face.obj'
            )
            tar_surface.add_surface_(verts.to(device=device), faces.to(device=device))

        try:
            verts, faces = io.ReadOBJ(source_surface_path)
            src_surface = core.TriangleMesh(verts, faces)
            src_surface.to_(device)
            src_surface.flip_normals_()
        except IOError:
            print(f'The raw head surface for {source_block} was not found ... Next block')
            continue

        extra_surfaces = []
        for path in extras_paths:
            try:
                verts, faces = io.ReadOBJ(path)
            except IOError:
                extra_name = path.split('/')[-1]
                print(f'{extra_name} not found as an extra ... removing from list')
                _ = extras_paths.pop(extras_paths.index(path))
                continue

            extra_surfaces += [core.TriangleMesh(verts, faces)]
            extra_surfaces[-1].to_(device)

        # Load or create the dictionary for registration
        try:
            with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_front_face_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'spatial_sigma': [2.0, 1.0],
                'affine_lr': 1.0e-06,
                'translation_lr': 1.0e-04,
                'converge': 0.01,
                'rigid_transform': True
            }

        print(f'Registering {source_block} to {target_block}:')

        affine_tform = front_face_register(
            tar_surface.copy(),
            src_surface.copy(),
            spatial_sigma=params['spatial_sigma'],
            affine_lr=params['affine_lr'],
            translation_lr=params['translation_lr'],
            converge=params['converge'],
            device=device
        )

        # Apply the affine to the source element and the excess
        aff_tformer = uo.AffineTransformSurface.Create(affine_tform, device=device)
        aff_src_surface = aff_tformer(src_surface)

        aff_extra_surface = []
        for surface in extra_surfaces:
            aff_extra_surface += [aff_tformer(surface)]

        out_path = f'{rabbit_dir}{source_block}{ff_ext}{source_block}'
        if not os.path.exists(f'{rabbit_dir}{source_block}{ff_ext}'):
            os.makedirs(f'{rabbit_dir}{source_block}{ff_ext}')

        if not os.path.exists(f'{out_path}_head_front_face.obj') or rerun:
            io.WriteOBJ(aff_src_surface.vertices, aff_src_surface.indices, f'{out_path}_head_front_face.obj')

        for extra_path, extra_surface in zip(extras_paths, aff_extra_surface):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_front_face.')
            if not os.path.exists(f'{out_path}{name}') or rerun:
                io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_front_face_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save the affine in the volumes and in the surfaces location
        np.savetxt(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_front_face_tform.txt', affine_tform.numpy())

    print('Done registering foots blocks to middle block.')

    # Loop over the head blocks
    for i, block_path in enumerate(head_blocks, 1):

        if i == len(head_blocks):
            break

        target_block = block_path.split('/')[-1]
        source_block = head_blocks[i].split('/')[-1]

        if source_block in skip_blocks:
            continue

        if os.path.exists(f'{rabbit_dir}{source_block}{stitch_ext}'):
            mid_path = f'{source_block}{stitch_ext}'
        else:
            mid_path = f'{source_block}{raw_ext}'

        target_surface_path = f'{rabbit_dir}{target_block}{ff_ext}{target_block}_head_front_face.obj'
        source_surface_path = f'{rabbit_dir}{mid_path}{source_block}_foot.obj'

        extras_paths = [
            f'{rabbit_dir}{mid_path}{source_block}_decimate.obj',
            f'{rabbit_dir}{mid_path}{source_block}_ext.obj'
        ]

        if i < len(head_blocks) - 1:
            extras_paths += [f'{rabbit_dir}{mid_path}{source_block}_head.obj']

        if os.path.exists(f'{rabbit_dir}{mid_path}{source_block}_head_support.obj'):
            extras_paths += [f'{rabbit_dir}{mid_path}{source_block}_head_support.obj']

        if os.path.exists(
                f'{rabbit_dir}{source_block}{ff_ext}{source_block}_foot_front_face.obj') and not rerun:
            print(f'The front face surface for {source_block} already exists ... Next block')
            continue

        try:
            verts, faces = io.ReadOBJ(target_surface_path)
            tar_surface = core.TriangleMesh(verts, faces)
            tar_surface.to_(device)
        except IOError:
            print(f'The front face foot surface for {target_block} was not found ... Next block')
            continue

        # Need to see if the target needs any support
        support_block = block_list[block_list.index(block_path) + 1].split('/')[-1]
        if os.path.exists(f'{rabbit_dir}{support_block}{ff_ext}{support_block}_head_support_front_face.obj'):
            verts, faces = io.ReadOBJ(
                f'{rabbit_dir}{support_block}{ff_ext}{support_block}_head_support_front_face.obj'
            )
            tar_surface.add_surface_(verts.to(device=device), faces.to(device=device))

        try:
            verts, faces = io.ReadOBJ(source_surface_path)
            src_surface = core.TriangleMesh(verts, faces)
            src_surface.to_(device)
            src_surface.flip_normals_()
        except IOError:
            print(f'The raw foot surface for {source_block} was not found ... Next block')
            continue

        extra_surfaces = []
        for path in extras_paths:
            try:
                verts, faces = io.ReadOBJ(path)
            except IOError:
                extra_name = path.split('/')[-1]
                print(f'{extra_name} not found as an extra ... removing from list')
                _ = extras_paths.pop(extras_paths.index(path))
                continue

            extra_surfaces += [core.TriangleMesh(verts, faces)]
            extra_surfaces[-1].to_(device)

        # Load or create the dictionary for registration
        try:
            with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_front_face_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'spatial_sigma': [2.0, 1.0],
                'affine_lr': 1.0e-06,
                'translation_lr': 1.0e-04,
                'converge': 0.01,
                'rigid_transform': True
            }

        print(f'Registering {source_block} to {target_block}:')

        affine_tform = front_face_register(
            tar_surface.copy(),
            src_surface.copy(),
            spatial_sigma=params['spatial_sigma'],
            affine_lr=params['affine_lr'],
            translation_lr=params['translation_lr'],
            converge=params['converge'],
            device=device
        )

        # Apply the affine to the source element and the excess
        aff_tformer = uo.AffineTransformSurface.Create(affine_tform, device=device)
        aff_src_surface = aff_tformer(src_surface)

        aff_extra_surface = []
        for surface in extra_surfaces:
            aff_extra_surface += [aff_tformer(surface)]

        out_path = f'{rabbit_dir}{source_block}{ff_ext}{source_block}'
        if not os.path.exists(f'{rabbit_dir}{source_block}{ff_ext}'):
            os.makedirs(f'{rabbit_dir}{source_block}{ff_ext}')

        if not os.path.exists(f'{out_path}_foot_front_face.obj') or rerun:
            io.WriteOBJ(aff_src_surface.vertices, aff_src_surface.indices, f'{out_path}_foot_front_face.obj')

        for extra_path, extra_surface in zip(extras_paths, aff_extra_surface):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_front_face.')
            if not os.path.exists(f'{out_path}{name}') or rerun:
                io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_front_face_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save the affine in the volumes and in the surfaces location
        np.savetxt(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_front_face_tform.txt', affine_tform.numpy())

    print('Done registering head blocks to middle block.')


if __name__ == '__main__':
    rabbit = '18_047'
    # process_mic(rabbit)
    # match_bf_mic()
    front_face_stacking(rabbit)
