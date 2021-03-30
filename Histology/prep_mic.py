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


def process_mic(rabbit):

    raw_mic_dir = f'/hdscratch/ucair/{rabbit}/microscopic/'
    bf_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_bf_dir = f'/hdscratch/ucair/blockface/{rabbit}/'

    block_list = sorted(glob.glob(f'{raw_mic_dir}/*'))

    # for block_path in block_list:
    #     block = block_path.split('/')[-1]
    #
    #     mic_list = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))
    #
    #     img_nums = [x.split('/')[-1].split('_')[1] for x in mic_list]
    #
    #     # Load the image
    #     for img in img_nums:
    #         print(f'Processing {block}, {img} ... ', end='')
    #         mic_file = f'{raw_mic_dir}{block}/hdf5/{block}_img{img}_image.hdf5'
    #
    #         mic = io.LoadITKFile(f'{raw_mic_dir}{block}/raw/IMG_{img}_histopathology_image.tif')
    #
    #         with h5py.File(mic_file, 'w') as f:
    #             g = f.create_group('RawImage')
    #             g.create_dataset('ImageData', data=mic.data.numpy())
    #
    #         print('Done')

    for block_path in block_list:
        block = block_path.split('/')[-1]

        mic_list = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))

        img_nums = [x.split('/')[-1].split('_')[1] for x in mic_list]

        for img in img_nums:

            mic_file = f'{raw_mic_dir}{block}/hdf5/{block}_img{img}_image.hdf5'
            mic_seg = f'{raw_mic_dir}{block}/hdf5/{block}_img{img}_label.hdf5'
            blockface_image = f'{bf_dir}{block}/volumes/raw/difference/IMG_{img}_difference.mhd'
            blockface_label = f'{bf_dir}{block}/volumes/raw/segmentation/IMG_{img}_segmentation.mhd'

            meta_dict = {}
            with h5py.File(mic_file, 'r') as f:
                for key in f['RawImage'].attrs:
                    meta_dict[key] = f['RawImage'].attrs[key]

            if 'Affine' in meta_dict.keys():
                continue

            blockface = io.LoadITKFile(blockface_image, device=device)
            label_data = io.LoadITKFile(blockface_label, device=device)

            blockface = (blockface - blockface.min()) / (blockface.max() - blockface.min())
            label = blockface.clone()
            label.data = label_data.data.clone()

            print(f'Affine Registering ... ')
            aff_mic, aff_mic_seg, affine = tools.process_mic(mic_file, mic_seg, blockface, label, device=device)
            print(f'Done')

            aff_mic *= aff_mic_seg
            blockface *= label

            blockface_s = core.StructuredGrid.FromGrid(blockface, tensor=blockface[0].unsqueeze(0), channels=1)
            aff_mic = (aff_mic - aff_mic.min()) / (aff_mic.max() - aff_mic.min())

            print(f'Deformable Registering Labels ... ')
            def_label, label_deformation = match_bf_mic(
                aff_mic_seg,
                label,
                steps=[0.01, 0.005],
                scales=[4, 1],
                gauss=True
            )
            print(f'Done')

            label_def_mic = so.ApplyGrid(label_deformation, device=device)(aff_mic, label_deformation)

            print(f'Deformable Registering Images ... ')
            def_image, image_deformation = match_bf_mic(
                label_def_mic,
                blockface_s,
                steps=[0.01, 0.01],
                scales=[2, 1],
            )
            print(f'Done')

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

            print('Saving ... ')

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

            print('Done')

            plt.close('all')

    print('All Done')


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

    # plt.figure()
    # plt.imshow(source_def[0].cpu().squeeze(), cmap='gray')
    # plt.colorbar()
    # plt.title('Deformed Image')
    # plt.figure()
    # plt.imshow(source_def[0].cpu().squeeze() - match.target[0].cpu().squeeze(), cmap='gray')
    # plt.colorbar()
    # plt.title('Difference With Target')

    return source_def, deformation


if __name__ == '__main__':
    rabbit = '18_047'
    process_mic(rabbit)
