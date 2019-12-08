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
    skip_blocks = ['block02', 'block03', 'block04', 'block05', 'block08', 'block09', 'block10', 'block11', 'block12']

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

        extras_paths = [
            f'{rabbit_dir}{source_block}{raw_ext}{source_block}_decimate.obj',
            f'{rabbit_dir}{source_block}{raw_ext}{source_block}_ext.obj'
        ]

        if i < len(foot_blocks) - 1:
            extras_paths += [f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot.obj']

        if os.path.exists(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot_support.obj'):
            extras_paths += [f'{rabbit_dir}{source_block}{raw_ext}{source_block}_foot_support.obj']

        if os.path.exists(f'{rabbit_dir}{source_block}{deform_ext}{source_block}_head_deformable.obj') and not rerun:
            print(f'The deformed surface for {source_block} already exists ... Next block')
            continue

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
                'spatial_sigma': [5.0, 0.5],
                'smoothing_sigma': [1.0, 1.0, 10.0],
                'deformable_lr': 1.0e-04,
                'converge': 0.3,
                'rigid_transform': True,
                'phi_inv_size': [25, 128, 128]
            }

        # Do the deformable registration
        def_surface, def_extras, phi_inv = tools.deformable_register(
            tar_surface.copy(),
            aff_src_surface.copy(),
            spatial_sigma=params['spatial_sigma'],
            deformable_lr=params['deformable_lr'],
            phi_inv_size=params['phi_inv_size'],
            src_excess=aff_extra_surface,
            device=device
        )

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_deformable_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save out all of the deformable transformed surfaces and phi inv
        io.SaveITKFile(phi_inv, f'{rabbit_dir}{source_block}{vol_ext}{source_block}_phi_inv.mhd')
        out_path = f'{rabbit_dir}{source_block}{deform_ext}{source_block}'
        if not os.path.exists(f'{out_path}_head_deformable.obj') or rerun:
            io.WriteOBJ(def_surface.vertices, def_surface.indices, f'{out_path}_head_deformable.obj')

        for extra_path, extra_surface in zip(extras_paths, def_extras):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_deformable.')
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
                'spatial_sigma': [5.0, 0.5],
                'smoothing_sigma': [1.0, 1.0, 10.0],
                'deformable_lr': 1.0e-04,
                'converge': 0.3,
                'rigid_transform': True,
                'phi_inv_size': [20, 96, 96]
            }

        # Do the deformable registration
        def_surface, def_extras, phi_inv = tools.deformable_register(
            tar_surface.copy(),
            aff_src_surface.copy(),
            spatial_sigma=params['spatial_sigma'],
            deformable_lr=params['deformable_lr'],
            phi_inv_size=params['phi_inv_size'],
            src_excess=aff_extra_surface,
            device=device
        )

        # Save out the parameters:
        with open(f'{rabbit_dir}{source_block}{raw_ext}{source_block}_deformable_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save out all of the deformable transformed surfaces and phi inv
        io.SaveITKFile(phi_inv, f'{rabbit_dir}{source_block}{vol_ext}{source_block}_phi_inv.mhd')
        out_path = f'{rabbit_dir}{source_block}{deform_ext}{source_block}'
        if not os.path.exists(f'{out_path}_foot_deformable.obj') or rerun:
            io.WriteOBJ(def_surface.vertices, def_surface.indices, f'{out_path}_foot_deformable.obj')

        for extra_path, extra_surface in zip(extras_paths, def_extras):
            name = extra_path.split('/')[-1].split(f'{source_block}')[-1].replace('.', '_deformable.')
            if not os.path.exists(f'{out_path}{name}') or rerun:
                io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

    print('Done registering head blocks to middle block.')


def match_bf_mic(mic_seged, blockface_seged, scales, steps, gauss=False):
    # ds_blockface = so.ResampleWorld.Create(microscopic, device=device)(blockface)
    # ds_blockface = (ds_blockface - ds_blockface.min()) / (ds_blockface.max() - ds_blockface.min())

    # Can only do 1 channel
    # microscopic.data = microscopic.data[0].unsqueeze(0)
    # microscopic.channels = 1
    # ds_blockface.data = ds_blockface.data[0].unsqueeze(0)
    # ds_blockface.channels = 1

    deformation = blockface_seged.clone()
    deformation.set_to_identity_lut_()
    deformation_list = []

    # Create a grid composer
    composer = so.ComposeGrids(device=device, dtype=torch.float32, padding_mode='border')

    if gauss:
        # Need do some blurring for the mic
        gauss = so.Gaussian.Create(
            channels=1,
            kernel_size=25,
            sigma=10,
            device=device
        )
        #
        mic_seged = gauss(mic_seged)

    # Now need to variance equalize
    # veq = so.VarianceEqualize.Create(
    #     kernel_size=20,
    #     sigma=10,
    #     device=device
    # )
    #
    # ve_mic = veq(mic_seged.clone())
    # ve_block = veq(blockface_seged)

    # Steps
    for s in scales:

        temp_mic = mic_seged.clone()
        temp_block = blockface_seged.clone()

        scale_source = temp_mic.set_size(mic_seged.size // s, inplace=False)
        scale_target = temp_block.set_size(blockface_seged.size // s, inplace=False)
        deformation = deformation.set_size(blockface_seged.size // s, inplace=False)

        # Apply the deformation to the source image
        scale_source = so.ApplyGrid(deformation)(scale_source)

        operator = so.FluidKernel.Create(
            scale_target,
            device=device,
            alpha=1.0,
            beta=0.0,
            gamma=0.001,
        )

        similarity = so.L2Similarity.Create(dim=2, device=device)

        match = st.IterativeMatch.Create(
            source=scale_source,
            target=scale_target,
            similarity=similarity,
            operator=operator,
            device=device,
            step_size=steps[scales.index(s)],
            incompressible=False
        )

        energy = [match.initial_energy]
        print(f'Iteration: 0   Energy: {match.initial_energy}')
        for i in range(1, 500):
            energy.append(match.step())
            print(f'Iteration: {i}   Energy: {energy[-1]}')

            if i > 10 and np.mean(energy[-10:]) - energy[-1] < 0.001:
                break

        deformation = match.get_field()
        deformation_list.append(deformation.clone().set_size(mic_seged.size, inplace=False))
        deformation = composer(deformation_list[::-1])

    # Compose the deformation fields
    source_def = so.ApplyGrid(deformation, device=device)(mic_seged, deformation)

    plt.figure()
    plt.imshow(source_def[0].cpu().squeeze(), cmap='gray')
    plt.colorbar()
    plt.title('Deformed Image')
    plt.figure()
    plt.imshow(source_def[0].cpu().squeeze() - match.target[0].cpu().squeeze(), cmap='gray')
    plt.colorbar()
    plt.title('Difference With Target')

    return source_def, deformation


def process_mic(rabbit):

    raw_mic_dir = f'/hdscratch/ucair/{rabbit}/microscopic/'
    raw_bf_dir = f'/hdscratch/ucair/blockface/{rabbit}/'

    block_list = sorted(glob.glob(f'{raw_mic_dir}/*'))

    for block_path in block_list[10:]:
        block = block_path.split('/')[-1]

        mic_list = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))

        img_nums = [x.split('/')[-1].split('_')[1] for x in mic_list]

        for img in img_nums[3:]:

            mic_file = f'{raw_mic_dir}{block}/hdf5/{block}_img{img}_image.hdf5'
            mic_seg = f'{raw_mic_dir}{block}/hdf5/{block}_img{img}_label.hdf5'
            blockface_image = f'{raw_bf_dir}affineImages/{block}/difference/IMG_{img}_difference.mhd'
            blockface_label = f'{raw_bf_dir}labels_hd/{block}/IMG_{img}_label_hd.nrrd'

            blockface = io.LoadITKFile(blockface_image, device=device)
            label_data = io.LoadITKFile(blockface_label, device=device)

            blockface = (blockface - blockface.min()) / (blockface.max() - blockface.min())
            label = blockface.clone()
            label.data = label_data.data.clone()

            aff_mic, aff_mic_seg, affine = tools.process_mic(mic_file, mic_seg, blockface, label, device=device)

            aff_mic *= aff_mic_seg
            blockface *= label

            blockface_s = core.StructuredGrid.FromGrid(blockface, tensor=blockface[0].unsqueeze(0), channels=1)
            aff_mic = (aff_mic - aff_mic.min()) / (aff_mic.max() - aff_mic.min())

            def_label, label_deformation = match_bf_mic(
                aff_mic_seg,
                label,
                steps=[0.01, 0.005],
                scales=[4, 1],
                gauss=True
            )

            label_def_mic = so.ApplyGrid(label_deformation, device=device)(aff_mic, label_deformation)

            def_image, image_deformation = match_bf_mic(
                label_def_mic,
                blockface_s,
                steps=[0.01, 0.01],
                scales=[2, 1],
            )

            composer = so.ComposeGrids(device=device, dtype=torch.float32, padding_mode='border')
            deformation = composer([image_deformation, label_deformation])

            def_mic = so.ApplyGrid(deformation, device=device)(aff_mic, deformation)

            try:
                with h5py.File(mic_file, 'r') as f:
                    mic = f['ImageData'][:]
            except KeyError:
                with h5py.File(mic_file, 'r') as f:
                    mic = f['RawImage/ImageData'][:]

            mic = core.StructuredGrid(
                mic.shape[1:],
                tensor=torch.tensor(mic, dtype=torch.float32, device=device),
                device=device,
                dtype=torch.float32,
                channels=3
            )

            with h5py.File(mic_file, 'w') as f:
                g = f.create_group('RawImage')
                d = f.create_group('Deformation')
                g.create_dataset('ImageData', data=mic.data.cpu().numpy())
                d.create_dataset('Phi', data=deformation.data.cpu().numpy())
                g.attrs['Shape'] = list(mic.shape())
                g.attrs['Spacing'] = mic.spacing.tolist()
                g.attrs['Origin'] = mic.origin.tolist()
                g.attrs['Affine'] = affine.tolist()
                d.attrs['Shape'] = list(deformation.shape())
                d.attrs['Spacing'] = deformation.spacing.tolist()
                d.attrs['Origin'] = deformation.origin.tolist()

            io.SaveITKFile(def_mic, f'{raw_mic_dir}{block}/volume/images/IMG_{img}_def_histopathology.mhd')

            plt.close('all')

    print('All Done')


if __name__ == '__main__':
    rabbit = '18_047'
    # process_mic(rabbit)
    # match_bf_mic()
    block_stacking(rabbit)
