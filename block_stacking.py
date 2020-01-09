import os
import sys
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

device = 'cuda:0'

# with open(f'{data_dir}{rabbit}_config.yaml') as f:
#     info = yaml.load(f, Loader=yaml.FullLoader)
#
# with open(f'FILE', 'w') as f:
#     yaml.dump(DATA, f)


def block_stacking(rabbit):

    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_ext = '/surfaces/raw/'
    rigid_ext = '/surfaces/rigid/'
    deform_ext = '/surfaces/deformable/'
    vol_ext = '/volumes/raw/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    # Determine the middle block
    middle_block = block_list[len(block_list) // 2]
    foot_blocks = block_list[block_list.index(middle_block):]
    head_blocks = block_list[:block_list.index(middle_block) + 1][::-1]

    rerun = True
    skip_blocks = []

    # for i, block_path in enumerate(block_list):
    #
    #     block = block_path.split('/')[-1]
    #     target_surface_path = f'{rabbit_dir}{block}{raw_ext}{block}_target_piece_surface.obj'
    #     source_surface_path = f'{rabbit_dir}{block}{raw_ext}{block}_source_piece_surface.obj'
    #
    #     if block == 'block07':
    #         continue
    #
    #     try:
    #         verts, faces = io.ReadOBJ(target_surface_path)
    #         tar_surface = core.TriangleMesh(verts, faces)
    #         tar_surface.to_(device)
    #     except IOError:
    #         print(f'The target stitching surface for {block} was not found ... skipping')
    #         continue
    #
    #     try:
    #         verts, faces = io.ReadOBJ(source_surface_path)
    #         src_surface = core.TriangleMesh(verts, faces)
    #         src_surface.to_(device)
    #         src_surface.flip_normals_()
    #     except IOError:
    #         print(f'The source stitching surface for {block} was not found ... skipping')
    #         continue
    #
    #     extras_paths = [
    #         f'{rabbit_dir}{block}{raw_ext}{block}_source_piece_ext.obj'
    #     ]
    #
    #     # if i == 0:
    #     #     extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_foot.obj']
    #     # elif i == len(block_list) - 1:
    #     #     extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_head.obj']
    #     # else:
    #     #     extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_foot.obj']
    #     #     extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_head.obj']
    #
    #     extra_surfaces = []
    #     for path in extras_paths:
    #         try:
    #             verts, faces = io.ReadOBJ(path)
    #         except IOError:
    #             extra_name = path.split('/')[-1]
    #             print(f'{extra_name} not found as an extra ... removing from list')
    #             _ = extras_paths.pop(extras_paths.index(path))
    #             continue
    #
    #         extra_surfaces += [core.TriangleMesh(verts, faces)]
    #         extra_surfaces[-1].to_(device)
    #     # Load or create the dictionary for registration
    #     try:
    #         with open(f'{rabbit_dir}{block}{raw_ext}{block}_stitch_config.yaml', 'r') as f:
    #             params = yaml.load(f, Loader=yaml.FullLoader)
    #     except IOError:
    #         params = {
    #             'spatial_sigma': [1.0, 0.05],
    #             'smoothing_sigma': [10.0, 10.0, 0.1],
    #             'deformable_lr': [1.0e-04, 1.0e-04],
    #             'converge': 0.01,
    #             'rigid_transform': True,
    #             'phi_inv_size': [25, 128, 128]
    #         }
    #     # Do the deformable registration
    #     def_src_surface, def_extras, phi_inv = tools.deformable_register(
    #         tar_surface.copy(),
    #         src_surface.copy(),
    #         deformable_lr=0.001,
    #         spatial_sigma=params['spatial_sigma'],
    #         phi_inv_size=params['phi_inv_size'],
    #         smoothing_sigma=params['smoothing_sigma'],
    #         src_excess=extra_surfaces,
    #         converge=params['converge'],
    #         device=device
    #     )
    #
    #     # Save out the parameters:
    #     with open(f'{rabbit_dir}{block}{raw_ext}{block}_stitch_config.yaml', 'w') as f:
    #         yaml.dump(params, f)
    #
    #     # Save out all of the deformable transformed surfaces and phi inv
    #     io.SaveITKFile(phi_inv, f'{rabbit_dir}{block}{vol_ext}{block}_phi_inv_stitch.mhd')
    #     out_path = f'{rabbit_dir}{block}{raw_ext}{block}'
    #     # if not os.path.exists(f'{out_path}_head_deformable.obj') or rerun:
    #     io.WriteOBJ(def_src_surface.vertices, def_src_surface.indices, f'{out_path}_source_piece_surface_stitch.obj')
    #
    #     for extra_path, extra_surface in zip(extras_paths, def_extras):
    #         name = extra_path.split('/')[-1].split(f'{block}')[-1].replace('.', '_stitch.')
    #         if not os.path.exists(f'{out_path}{name}') or rerun:
    #             io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

    # Loop over the foot blocks
    for i, block_path in enumerate(foot_blocks, 1):

        if i == len(foot_blocks):
            break

        target_block = block_path.split('/')[-1]
        source_block = foot_blocks[i].split('/')[-1]

        if source_block in skip_blocks:
            continue

        target_surface_path = f'{rabbit_dir}{target_block}{deform_ext}{target_block}_foot_deformable.obj'
        source_surface_path = f'{rabbit_dir}{source_block}{raw_ext}{source_block}_head.obj'
        if os.path.exists(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_head_stitched.obj'):
            source_surface_path = f'{rabbit_dir}{source_block}{raw_ext}{source_block}_head_stitched.obj'

        extras_paths = [
            f'{rabbit_dir}{source_block}{raw_ext}{source_block}_decimate.obj',
            f'{rabbit_dir}{source_block}{raw_ext}{source_block}_ext.obj',
        ]

        if i < len(foot_blocks) - 1:
            extras_paths += [f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot.obj']

        if os.path.exists(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot_support.obj'):
            extras_paths += [f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot_support.obj']

        if os.path.exists(f'{rabbit_dir}{source_block}{deform_ext}{source_block}_head_deformable.obj') and not rerun:
            print(f'The deformed surface for {source_block} already exists ... Next block')
            continue

        # Need to check if there are stitched surfaces
        for i, path in enumerate(extras_paths):
            stitch_name = path.split('/')[-1].split('.')[0] + '_stitched.obj'
            if os.path.exists(f'{rabbit_dir}{source_block}{raw_ext}{stitch_name}'):
                extras_paths[i] = f'{rabbit_dir}{source_block}{raw_ext}{stitch_name}'

        try:
            verts, faces = io.ReadOBJ(target_surface_path)
            tar_surface = core.TriangleMesh(verts, faces)
            tar_surface.to_(device)
        except IOError:
            print(f'The deformed foot surface for {target_block} was not found ... Next block')
            continue

        # Need to see if the target needs any support
        support_block = block_list[block_list.index(block_path) - 1].split('/')[-1]
        if os.path.exists(f'{rabbit_dir}{support_block}{deform_ext}{support_block}_foot_support_deformable.obj'):
            verts, faces = io.ReadOBJ(
                f'{rabbit_dir}{support_block}{deform_ext}{support_block}_foot_support_deformable.obj'
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
            with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_affine_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'spatial_sigma': [2.0, 0.5],
                'affine_lr': 1.0e-06,
                'translation_lr': 1.0e-04,
                'converge': 0.01,
                'rigid_transform': True
            }

        print(f'Registering {source_block} to {target_block}:')

        affine_tform = tools.affine_register(
            tar_surface.copy(),
            src_surface.copy(),
            spatial_sigma=params['spatial_sigma'],
            affine_lr=params['affine_lr'],
            translation_lr=params['translation_lr'],
            rigid=params['rigid_transform'],
            converge=params['converge'],
            device=device
        )

        # Apply the affine to the source element and the excess
        aff_tformer = uo.AffineTransformSurface.Create(affine_tform, device=device)
        aff_src_surface = aff_tformer(src_surface)

        aff_extra_surface = []
        for surface in extra_surfaces:
            aff_extra_surface += [aff_tformer(surface)]

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_affine_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save out all of the affine transformed surfaces and the transformation
        out_path = f'{rabbit_dir}{source_block}{rigid_ext}{source_block}'

        # Save the affine in the volumes and in the surfaces location
        np.savetxt(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_rigid_tform.txt', affine_tform.numpy())
        np.savetxt(f'{rabbit_dir}{source_block}{vol_ext}{source_block}_rigid_tform.txt', affine_tform.numpy())

        if not os.path.exists(f'{out_path}_head_rigid.obj') or rerun:
            io.WriteOBJ(aff_src_surface.vertices, aff_src_surface.indices, f'{out_path}_head_rigid.obj')

        for extra_path, extra_surface in zip(extras_paths, aff_extra_surface):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_rigid.')
            if not os.path.exists(f'{out_path}{name}') or rerun:
                io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

        try:
            with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_deformable_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [5.0, 0.5],
                'propagation_sigma': [1.0, 1.0, 10.0],
                'deformable_lr': [1.0e-04],
                'converge': 0.3,
                'grid_size': [25, 128, 128]
            }

        if 'spatial_sigma' in params.keys():
            params['currents_sigma'] = params['spatial_sigma']
            del params['spatial_sigma']
        if 'phi_inv_size' in params.keys():
            params['grid_size'] = params['phi_inv_size']
            del params['phi_inv_size']
        if 'rigid_transform' in params.keys():
            del params['rigid_transform']
        if 'smoothing_sigma' in params.keys():
            params['propagation_sigma'] = params['smoothing_sigma']
            del params['smoothing_sigma']

        if type(params['deformable_lr']) is not list:
            params['deformable_lr'] = [params['deformable_lr']] * len(params['spatial_sigma'])

        # Do the deformable registration
        def_surface, def_extras, phi, phi_inv = tools.deformable_register(
            tar_surface.copy(),
            aff_src_surface.copy(),
            currents_sigma=params['currents_sigma'],
            prop_sigma=params['propagation_sigma'],
            deformable_lr=params['deformable_lr'],
            converge=params['converge'],
            grid_size=params['grid_size'],
            src_excess=aff_extra_surface,
            accu_forward=True,
            accu_inverse=True,
            device=device,
            grid_device='cuda:1'
        )

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_deformable_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save out all of the deformable transformed surfaces and phi inv
        io.SaveITKFile(phi_inv, f'{rabbit_dir}{source_block}{vol_ext}{source_block}_phi_inv_stacking.mhd')
        io.SaveITKFile(phi, f'{rabbit_dir}{source_block}{vol_ext}{source_block}_phi_stacking.mhd')
        out_path = f'{rabbit_dir}{source_block}{deform_ext}{source_block}'
        if not os.path.exists(f'{out_path}_head_deformable.obj') or rerun:
            io.WriteOBJ(def_surface.vertices, def_surface.indices, f'{out_path}_head_deformable.obj')

        for extra_path, extra_surface in zip(extras_paths, def_extras):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_deformable.')
            if '_stitched' in name:
                name.replace('_stitched', '')
            if not os.path.exists(f'{out_path}{name}') or rerun:
                io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

    print('Done registering foots blocks to middle block.')

    # Loop over the head blocks
    for i, block_path in enumerate(head_blocks, 1):

        if i == len(head_blocks):
            break

        target_block = block_path.split('/')[-1]
        source_block = head_blocks[i].split('/')[-1]

        if source_block in skip_blocks:
            continue

        target_surface_path = f'{rabbit_dir}{target_block}{deform_ext}{target_block}_head_deformable.obj'
        source_surface_path = f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot.obj'
        if os.path.exists(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot_stitched.obj'):
            source_surface_path = f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot_stitched.obj'

        extras_paths = [
            f'{rabbit_dir}{source_block}{raw_ext}{source_block}_decimate.obj',
            f'{rabbit_dir}{source_block}{raw_ext}{source_block}_ext.obj'
        ]

        if i < len(head_blocks) - 1:
            extras_paths += [f'{rabbit_dir}{source_block}{raw_ext}{source_block}_head.obj']

        if os.path.exists(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_head_support.obj'):
            extras_paths += [f'{rabbit_dir}{source_block}{raw_ext}{source_block}_head_support.obj']

        if os.path.exists(
                f'{rabbit_dir}{source_block}{deform_ext}{source_block}_foot_deformable.obj') and not rerun:
            print(f'The deformed surface for {source_block} already exists ... Next block')
            continue

        # Need to check if there are stitched surfaces
        for i, path in enumerate(extras_paths):
            stitch_name = path.split('/')[-1].split('.')[0] + '_stitched.obj'
            if os.path.exists(f'{rabbit_dir}{source_block}{raw_ext}{stitch_name}'):
                extras_paths[i] = f'{rabbit_dir}{source_block}{raw_ext}{stitch_name}'

        try:
            verts, faces = io.ReadOBJ(target_surface_path)
            tar_surface = core.TriangleMesh(verts, faces)
            tar_surface.to_(device)
        except IOError:
            print(f'The deformed foot surface for {target_block} was not found ... Next block')
            continue

        # Need to see if the target needs any support
        support_block = block_list[block_list.index(block_path) + 1].split('/')[-1]
        if os.path.exists(f'{rabbit_dir}{support_block}{deform_ext}{support_block}_head_support_deformable.obj'):
            verts, faces = io.ReadOBJ(
                f'{rabbit_dir}{support_block}{deform_ext}{support_block}_head_support_deformable.obj'
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
            with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_affine_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'spatial_sigma': [2.0, 0.5],
                'affine_lr': 1.0e-06,
                'translation_lr': 1.0e-04,
                'converge': 0.01,
                'rigid_transform': True
            }

        print(f'Registering {source_block} to {target_block}:')

        affine_tform = tools.affine_register(
            tar_surface.copy(),
            src_surface.copy(),
            spatial_sigma=params['spatial_sigma'],
            affine_lr=params['affine_lr'],
            translation_lr=params['translation_lr'],
            rigid=params['rigid_transform'],
            converge=params['converge'],
            device=device
        )

        # Apply the affine to the source element and the excess
        aff_tformer = uo.AffineTransformSurface.Create(affine_tform, device=device)
        aff_src_surface = aff_tformer(src_surface)

        aff_extra_surface = []
        for surface in extra_surfaces:
            aff_extra_surface += [aff_tformer(surface)]

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_affine_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save out all of the affine transformed surfaces and the transformation
        out_path = f'{rabbit_dir}{source_block}{rigid_ext}{source_block}'

        # Save the affine in the volumes and in the surfaces location
        np.savetxt(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_rigid_tform.txt', affine_tform.numpy())
        np.savetxt(f'{rabbit_dir}{source_block}{vol_ext}{source_block}_rigid_tform.txt', affine_tform.numpy())

        if not os.path.exists(f'{out_path}_foot_rigid.obj') or rerun:
            io.WriteOBJ(aff_src_surface.vertices, aff_src_surface.indices, f'{out_path}_foot_rigid.obj')

        for extra_path, extra_surface in zip(extras_paths, aff_extra_surface):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_rigid.')
            if not os.path.exists(f'{out_path}{name}') or rerun:
                io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

        try:
            with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_deformable_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [5.0, 0.5],
                'propagation_sigma': [1.0, 1.0, 10.0],
                'deformable_lr': [1.0e-04],
                'converge': 0.3,
                'grid_size': [25, 128, 128]
            }

            if 'spatial_sigma' in params.keys():
                params['currents_sigma'] = params['spatial_sigma']
                del params['spatial_sigma']
            if 'phi_inv_size' in params.keys():
                params['grid_size'] = params['phi_inv_size']
                del params['phi_inv_size']
            if 'rigid_transform' in params.keys():
                del params['rigid_transform']
            if 'smoothing_sigma' in params.keys():
                params['propagation_sigma'] = params['smoothing_sigma']
                del params['smoothing_sigma']

            if type(params['deformable_lr']) is not list:
                params['deformable_lr'] = [params['deformable_lr']] * len(params['spatial_sigma'])

            # Do the deformable registration
            def_surface, def_extras, phi, phi_inv = tools.deformable_register(
                tar_surface.copy(),
                aff_src_surface.copy(),
                currents_sigma=params['currents_sigma'],
                prop_sigma=params['propagation_sigma'],
                deformable_lr=params['deformable_lr'],
                converge=params['converge'],
                grid_size=params['grid_size'],
                src_excess=aff_extra_surface,
                accu_forward=True,
                accu_inverse=True,
                device=device,
                grid_device='cuda:1'
            )

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_deformable_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save out all of the deformable transformed surfaces and phi inv
        io.SaveITKFile(phi_inv, f'{rabbit_dir}{source_block}{vol_ext}{source_block}_phi_inv_stacking.mhd')
        io.SaveITKFile(phi, f'{rabbit_dir}{source_block}{vol_ext}{source_block}_phi_stacking.mhd')
        out_path = f'{rabbit_dir}{source_block}{deform_ext}{source_block}'
        if not os.path.exists(f'{out_path}_foot_deformable.obj') or rerun:
            io.WriteOBJ(def_surface.vertices, def_surface.indices, f'{out_path}_foot_deformable.obj')

        for extra_path, extra_surface in zip(extras_paths, def_extras):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_deformable.')
            if '_stitched' in name:
                name.replace('_stitched', '')
            if not os.path.exists(f'{out_path}{name}') or rerun:
                io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

    print('Done registering head blocks to middle block.')


if __name__ == '__main__':
    rabbit = '18_047'
    # process_mic(rabbit)
    # match_bf_mic()
    block_stacking(rabbit)
