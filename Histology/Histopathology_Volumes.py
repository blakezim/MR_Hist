import os
import sys
sys.path.append("/home/sci/blakez/code/")
import glob
import yaml
import copy
import h5py
import tools
import torch
import numpy as np
import subprocess as sp
import skimage.segmentation as seg
from skimage import measure
from GenerateDeformation import generate

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

# device = 'cuda:1'
device='cpu'


def generate_image_volume(rabbit, block, base_dir='/hdscratch/ucair/'):
    blockface_dir = f'{base_dir}{rabbit}/blockface/{block}/'
    histology_dir = f'{base_dir}{rabbit}/microscopic/{block}/'
    def_dir = f'{histology_dir}deformations/'

    out_dir = f'{histology_dir}/volume/raw/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    raw_images = glob.glob(f'{histology_dir}/raw/*_image.tif')
    if not raw_images:
        raw_images += sorted(raw_images + glob.glob(f'{histology_dir}/raw/*_image.jpg'))
    image_nums = sorted([int(x.split('/')[-1].split('_')[1]) for x in raw_images])

    deformed_images = []

    for im in image_nums:

        print(f'Loading and deforming image {im} ... ', end='')

        # Load the healthy segmentation as a reference
        healthy = io.LoadITKFile(f'{histology_dir}segmentations/IMG_{im:03d}/img_{im:03d}_healthy_tissue.nrrd',
                                 device=device)
        try:
            ablation = io.LoadITKFile(f'{histology_dir}segmentations/IMG_{im:03d}/img_{im:03d}_ablated_region.nrrd',
                                      device=device)
        except RuntimeError:
            ablation = (healthy * 0.0).copy()

        try:
            transition = io.LoadITKFile(f'{histology_dir}segmentations/IMG_{im:03d}/img_{im:03d}_uncertain_region.nrrd',
                                        device=device)
        except RuntimeError:
            transition = (healthy * 0.0).copy()

        combined = ablation + transition + healthy
        combined.data = (combined.data >= 0.5).float()

        # Load the affine
        aff = np.loadtxt(glob.glob(f'{def_dir}img_{im:03d}_affine_to_blockface.txt')[0])
        aff = torch.tensor(aff, device=device, dtype=torch.float32)

        # Load the deformation
        deformation_data = io.LoadITKFile(f'{def_dir}/img_{im:03d}_deformation_to_blockface.mhd', device=device)

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

        deformation.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                        deformation.data.permute(list(range(1, 2 + 1)) + [0]).unsqueeze(-1))
        deformation.data = (deformation.data.squeeze() + t).permute([-1] + list(range(0, 2)))

        # Load the actual image
        #### Apply the affine to the image
        mic_file = f'{histology_dir}hdf5/{block}_img{im:03d}_image.hdf5'

        meta_dict = {}
        with h5py.File(mic_file, 'r') as f:
            mic = f['RawImage/ImageData'][:, ::10, ::10]
            for key in f['RawImage'].attrs:
                meta_dict[key] = f['RawImage'].attrs[key]

        mic = core.StructuredGrid(
            mic.shape[1:],
            tensor=torch.tensor(mic, dtype=torch.float32, device=device),
            spacing=torch.tensor([10.0, 10.0], dtype=torch.float32, device=device),
            device=device,
            dtype=torch.float32,
            channels=3
        )

        mic = (mic - mic.min()) / (mic.max() - mic.min())
        mic.data = mic.data * combined.data

        def_mic = so.ApplyGrid.Create(deformation, device=device)(mic, deformation)

        # Need to put the mic image in the right slice
        z_or = (im - 1) * 0.05
        def_mic = core.StructuredGrid(
            [2] + list(def_mic.shape()[1:]),
            tensor=def_mic.data.unsqueeze(1).repeat(1, 2, 1, 1),
            spacing=torch.tensor([0.05, def_mic.spacing[0], def_mic.spacing[1]], dtype=torch.float32, device=device),
            origin=torch.tensor([z_or, def_mic.origin[0], def_mic.origin[1]], dtype=torch.float32, device=device),
            device=device,
            dtype=torch.float32,
            channels=3
        )

        deformed_images.append(def_mic.copy())

        del def_mic, mic, deformation
        torch.cuda.empty_cache()

        print('done')

    # Now need to load the blockface volume
    block_vol = io.LoadITKFile(f'{blockface_dir}volumes/raw/difference_volume.mhd', device='cpu')
    # image_vol = core.StructuredGrid(
    #     size=block_vol.size,
    #     spacing=block_vol.spacing,
    #     origin=block_vol.origin,
    #     device=block_vol.device,
    #     channels=3
    # )
    image_z_spacing = np.diff(np.array(image_nums)).mean() * 0.05
    image_z_origin = image_nums[0] * 0.05

    image_vol = core.StructuredGrid(
        size=[len(image_nums)] + list(block_vol.size[-2:].numpy()),
        spacing=[image_z_spacing] + list(block_vol.spacing[-2:].numpy()),
        origin=[image_z_origin] + list(block_vol.origin[-2:].numpy()),
        device=block_vol.device,
        channels=3
    )

    for ii, im in enumerate(image_nums):
        image_vol.data[:, ii] = deformed_images[ii].data[:, 0].clone().cpu()

    io.SaveITKFile(image_vol, f'{out_dir}/{block}_image_volume.mhd')
    del image_vol, deformed_images, block_vol
    torch.cuda.empty_cache()


def generate_segmentation_volume(rabbit, block, base_dir='/hdscratch/ucair/'):
    blockface_dir = f'{base_dir}{rabbit}/blockface/{block}/'
    histology_dir = f'{base_dir}{rabbit}/microscopic/{block}/'
    def_dir = f'{histology_dir}deformations/'

    out_dir = f'{histology_dir}/volume/raw/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    raw_images = glob.glob(f'{histology_dir}/raw/*_image.tif')
    if not raw_images:
        raw_images += sorted(raw_images + glob.glob(f'{histology_dir}/raw/*_image.jpg'))
    image_nums = sorted([int(x.split('/')[-1].split('_')[1]) for x in raw_images])

    deformed_ablation_segs = []
    deformed_combined_segs = []
    deformed_images = []

    for im in image_nums:

        print(f'Loading and deforming image {im} ... ', end='')

        # Load the healthy segmentation as a reference
        healthy = io.LoadITKFile(f'{histology_dir}segmentations/IMG_{im:03d}/img_{im:03d}_healthy_tissue.nrrd',
                                 device=device)
        # Load the affine
        aff = np.loadtxt(glob.glob(f'{def_dir}img_{im:03d}_affine_to_blockface.txt')[0])
        aff = torch.tensor(aff, device=device, dtype=torch.float32)

        # Load the deformation
        deformation_data = io.LoadITKFile(f'{def_dir}/img_{im:03d}_deformation_to_blockface.mhd', device=device)

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

        deformation.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                        deformation.data.permute(list(range(1, 2 + 1)) + [0]).unsqueeze(-1))
        deformation.data = (deformation.data.squeeze() + t).permute([-1] + list(range(0, 2)))

        try:
            ablation = io.LoadITKFile(f'{histology_dir}segmentations/IMG_{im:03d}/img_{im:03d}_ablated_region.nrrd',
                                      device=device)
        except RuntimeError:
            ablation = (healthy * 0.0).copy()

        try:
            transition = io.LoadITKFile(f'{histology_dir}segmentations/IMG_{im:03d}/img_{im:03d}_uncertain_region.nrrd',
                                        device=device)
        except RuntimeError:
            transition = (healthy * 0.0).copy()

        # Load the actual image
        #### Apply the affine to the image
        mic_file = f'{histology_dir}hdf5/{block}_img{im:03d}_image.hdf5'

        meta_dict = {}
        with h5py.File(mic_file, 'r') as f:
            mic = f['RawImage/ImageData'][:, ::10, ::10]
            for key in f['RawImage'].attrs:
                meta_dict[key] = f['RawImage'].attrs[key]

        mic = core.StructuredGrid(
            mic.shape[1:],
            tensor=torch.tensor(mic, dtype=torch.float32, device=device),
            spacing=torch.tensor([10.0, 10.0], dtype=torch.float32, device=device),
            device=device,
            dtype=torch.float32,
            channels=3
        )

        mic = (mic - mic.min()) / (mic.max() - mic.min())

        if not os.path.exists(f'{histology_dir}deformable/img_{im:03d}_to_blockface/'):
            os.makedirs(f'{histology_dir}deformable/img_{im:03d}_to_blockface/')

        def_mic = so.ApplyGrid.Create(deformation, device=device)(mic, deformation)

        # Need to put the mic image in the right slice
        z_or = (im - 1) * 0.05
        def_mic = core.StructuredGrid(
            [2] + list(def_mic.shape()[1:]),
            tensor=def_mic.data.unsqueeze(1).repeat(1, 2, 1, 1),
            spacing=torch.tensor([0.05, def_mic.spacing[0], def_mic.spacing[1]], dtype=torch.float32, device=device),
            origin=torch.tensor([z_or, def_mic.origin[0], def_mic.origin[1]], dtype=torch.float32, device=device),
            device=device,
            dtype=torch.float32,
            channels=3
        )

        io.SaveITKFile(def_mic, f'{histology_dir}deformable/img_{im:03d}_to_blockface/img_{im:03d}_to_blockface.mhd')

        combined = ablation + transition
        combined.data = (combined.data >= 0.5).float()

        def_ablation = so.ApplyGrid.Create(deformation, device=device)(ablation, deformation)
        def_combined = so.ApplyGrid.Create(deformation, device=device)(combined, deformation)

        deformed_ablation_segs.append(def_ablation.copy())
        deformed_combined_segs.append(def_combined.copy())
        deformed_images.append(def_mic.copy())

        del def_ablation, def_combined, def_mic, combined, mic, deformation
        torch.cuda.empty_cache()

        print('done')

    # Now need to load the blockface volume
    block_vol = io.LoadITKFile(f'{blockface_dir}volumes/raw/difference_volume.mhd', device='cpu')
    image_vol = core.StructuredGrid(
        size=block_vol.size,
        spacing=block_vol.spacing,
        origin=block_vol.origin,
        device=block_vol.device,
        channels=3
    )

    for ii, im in enumerate(image_nums):
        image_vol.data[:, im - 1] = deformed_images[ii].data[:, 0].clone().cpu()

    io.SaveITKFile(image_vol, f'{out_dir}/{block}_image_volume.mhd')
    del image_vol, deformed_images
    torch.cuda.empty_cache()

    single_vol = core.StructuredGrid(
        size=block_vol.size,
        spacing=block_vol.spacing,
        origin=block_vol.origin,
        device=block_vol.device,
        channels=1
    )

    for ii, im in enumerate(image_nums):
        single_vol.data[:, im - 1] = deformed_ablation_segs[ii].data[:].clone().cpu()

    io.SaveITKFile(single_vol, f'{out_dir}/{block}_ablation_segmentation_no_interp.mhd')

    del single_vol
    torch.cuda.empty_cache()

    # Now need to load the blockface volume
    ablation_vol = core.StructuredGrid(
            size=block_vol.size,
            spacing=block_vol.spacing,
            origin=block_vol.origin,
            device=device,
            channels=1
    )

    for ii, im in enumerate(image_nums):

        if ii == len(image_nums) - 1:
            continue

        next_slice = image_nums[ii + 1]

        for s, slice in enumerate(range(im, next_slice + 1)):
            step = 1.0 / (next_slice - im)
            cur_alpha = 1.0 - (s * step)
            next_alpa = 0.0 + (s * step)

            ablation_vol.data[:, slice] = (deformed_ablation_segs[ii].data * cur_alpha) + (
                        deformed_ablation_segs[ii + 1].data * next_alpa)

    gauus_ablation = so.Gaussian.Create(1, 10, [0.1, 2, 2], dim=3, device=device)(ablation_vol)

    io.SaveITKFile(gauus_ablation, f'{out_dir}/{block}_ablation_segmentation.mhd')
    del gauus_ablation, deformed_ablation_segs
    torch.cuda.empty_cache()

    combined_vol = core.StructuredGrid(
        size=block_vol.size,
        spacing=block_vol.spacing,
        origin=block_vol.origin,
        device=block_vol.device,
        channels=1
    )

    for ii, im in enumerate(image_nums):

        if ii == len(image_nums) - 1:
            continue

        next_slice = image_nums[ii + 1]

        for s, slice in enumerate(range(im, next_slice + 1)):
            step = 1.0 / (next_slice - im)
            cur_alpha = 1.0 - (s * step)
            next_alpa = 0.0 + (s * step)

            combined_vol.data[:, slice] = (deformed_combined_segs[ii].data * cur_alpha) + (
                    deformed_combined_segs[ii + 1].data * next_alpa)

    # gauus_ablation = so.Gaussian.Create(1, 10, [0.1, 2, 2], dim=3, device=device)(ablation_vol)

    io.SaveITKFile(combined_vol, f'{out_dir}/{block}_ablation_and_transition_segmentation.mhd')

    del block_vol
    del combined_vol
    torch.cuda.empty_cache()


def deform_image_volumes(rabbit, block, base_dir='/hdscratch/ucair/'):

    histology_dir = f'{base_dir}{rabbit}/microscopic/{block}/'

    raw_dir = f'{histology_dir}/volume/raw/'
    out_dir = f'{histology_dir}/volume/deformable/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    deformation = generate(rabbit, block, source_space='blockface', target_space='invivo', base_dir=base_dir, pad_mode='zeros')

    def_file = f'/home/sci/blakez/ucair/longitudinal/{rabbit}/'
    def_file += 'deformation_fields/Day3_non_contrast_VIBE_interday_deformation_incomp.nii.gz'
    day0 = io.LoadITKFile(f'/hdscratch/ucair/AcuteBiomarker/Data/{rabbit}/{rabbit}_day3_to_day0_phi_inv.nii.gz', device=device)

    deformation = so.ComposeGrids.Create(device=device, padding_mode='zeros')([day0, deformation])

    deformation.to_(device=device)
    deformation.set_size((512, 512, 512))
    # Load the original volume
    image_vol = io.LoadITKFile(f'{raw_dir}/{block}_image_volume.mhd', device=device)
    def_image = so.ApplyGrid.Create(deformation, device=device)(image_vol, deformation)

    test_diff_z = np.diff(deformation.data[0, :].cpu().numpy())
    test_diff_y = np.diff(deformation.data[1, :].cpu().numpy())
    test_diff_x = np.diff(deformation.data[2, :].cpu().numpy())

    test_diff_z_small = np.abs(test_diff_z) >= 0.5
    test_diff_y_small = np.abs(test_diff_y) >= 0.5
    test_diff_x_small = np.abs(test_diff_x) >= 0.5

    mask = np.logical_and(np.logical_and(test_diff_x == 0, test_diff_y == 0), test_diff_z == 0)
    mask_small = np.logical_and(np.logical_and(test_diff_x_small  == 0, test_diff_y_small  == 0), test_diff_z_small  == 0)

    final_mask = np.logical_and(mask == False, mask_small == True)

    mask = torch.zeros_like(def_image.data)[0, None]
    mask[:, :, :, :-1] = torch.tensor(final_mask)

    def_image.data = def_image.data * mask

    io.SaveITKFile(def_image, f'{out_dir}/{block}_image_volume_to_invivo.mhd')


def deform_histology_volumes(rabbit, block, base_dir='/hdscratch/ucair/'):

    blockface_dir = f'{base_dir}{rabbit}/blockface/{block}/'
    histology_dir = f'{base_dir}{rabbit}/microscopic/{block}/'

    raw_dir = f'{histology_dir}/volume/raw/'
    out_dir = f'{histology_dir}/volume/deformable/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    deformation = generate(rabbit, block, source_space='blockface', target_space='invivo', base_dir=base_dir)
    deformation.to_(device=device)
    # Load the original volume
    ablation_vol = io.LoadITKFile(f'{raw_dir}/{block}_ablation_segmentation.mhd', device=device)
    ablation_vol.data = (ablation_vol.data >= 0.5).float()
    combined_vol = io.LoadITKFile(f'{raw_dir}/{block}_ablation_and_transition_segmentation.mhd', device=device)
    def_ablation = so.ApplyGrid.Create(deformation, device=device)(ablation_vol, deformation)
    def_combined = so.ApplyGrid.Create(deformation, device=device)(combined_vol, deformation)
    io.SaveITKFile(def_ablation, f'{out_dir}/{block}_ablation_segmentation_to_invivo.mhd')
    io.SaveITKFile(def_combined, f'{out_dir}/{block}_ablation_and_transition_segmentation_to_invivo.mhd')


def stack_hitology_volumes(rabbit, block_paths, base_dir='/hdscratch/ucair/'):

    def_ablations = []
    def_combineds = []
    out_dir = f'{base_dir}{rabbit}/microscopic/recons/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for block_path in block_paths:
        block = block_path.split('/')[-1]
        histology_dir = f'{base_dir}{rabbit}/microscopic/{block}/'
        def_dir = f'{histology_dir}/volume/deformable/'

        # Load the deformed volume
        if not os.path.exists(f'{def_dir}/{block}_ablation_segmentation_to_invivo.mhd'):
            continue
        def_ablation = io.LoadITKFile(f'{def_dir}/{block}_ablation_segmentation_to_invivo.mhd', device=device)
        def_combined = io.LoadITKFile(f'{def_dir}/{block}_ablation_and_transition_segmentation_to_invivo.mhd',
                                      device=device)

        # Threshold the volume
        def_ablation.data = (def_ablation.data >= 0.8).float()
        def_combined.data = (def_combined.data >= 0.8).float()

        def_ablations.append(def_ablation)
        def_combineds.append(def_combined)

    full_ablation = def_ablations[0].copy()
    full_ablation = full_ablation * 0.0
    for v in def_ablations:
        full_ablation = full_ablation + v

    full_ablation.data = (full_ablation.data > 0.0).float()

    io.SaveITKFile(full_ablation, f'{out_dir}all_ablation_segs_to_invivo.mhd')

    full_combined = def_ablations[0].copy()
    full_combined = full_combined * 0.0
    for v in def_combineds:
        full_combined = full_combined + v

    full_combined.data = (full_combined.data > 0.0).float()

    io.SaveITKFile(full_combined, f'{out_dir}all_ablation_and_transition_segs_to_invivo.mhd')


def stack_image_volumes(rabbit, block_paths, base_dir='/hdscratch/ucair/'):

    def_images = []
    out_dir = f'{base_dir}{rabbit}/microscopic/recons/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for block_path in block_paths:
        block = block_path.split('/')[-1]
        histology_dir = f'{base_dir}{rabbit}/microscopic/{block}/'
        def_dir = f'{histology_dir}/volume/deformable/'

        # Load the deformed volume
        if not os.path.exists(f'{def_dir}/{block}_image_volume_to_invivo.mhd'):
            continue
        def_image = io.LoadITKFile(f'{def_dir}/{block}_image_volume_to_invivo.mhd', device=device)

        def_images.append(def_image.copy())

    full_ablation = def_images[0].copy()
    full_ablation = full_ablation * 0.0
    for v in def_images:
        full_ablation.data = torch.stack([full_ablation.data, v.data], dim=0).max(0)[0]

    io.SaveITKFile(full_ablation, f'{out_dir}all_images_to_invivo.mhd')


if __name__ == '__main__':
    rabbit = '18_061'
    base_dir = '/hdscratch/ucair/'
    # block = 'block07'

    block_list = sorted(glob.glob(f'{base_dir}{rabbit}/microscopic/block*'))

    # input_vol = io.LoadITKFile('/hdscratch/ucair/18_047/mri/exvivo/volumes/raw/t2.nii.gz')
    # deformation = generate(rabbit, source_space='exvivo', target_space='blockface', block='block10', stitch=False)
    # deforemd_exvivo = so.ApplyGrid.Create(deformation)(input_vol, deformation)
    # io.SaveITKFile(deforemd_exvivo, '/home/sci/blakez/exvivo_to_block10.nrrd')
    #
    # input_vol = io.LoadITKFile('/hdscratch/ucair/18_047/blockface/block10/volumes/raw/difference_volume.mhd')
    # deformation = generate(rabbit, source_space='blockface', target_space='exvivo', block='block10', stitch=True)
    # deforemd_exvivo = so.ApplyGrid.Create(deformation)(input_vol, deformation)
    # io.SaveITKFile(deforemd_exvivo, '/home/sci/blakez/block10_to_exvivo.mhd')

    # from scipy.spatial.distance import directed_hausdorff
    # ablation_haus = io.ReadOBJ('/home/sci/blakez/hausdorff_hist.obj')
    # npv_haus = io.ReadOBJ('/home/sci/blakez/hausdorff_npv.obj')
    # directed_hausdorff(ablation_haus[0], npv_haus[0])

    # block = io.LoadITKFile('/hdscratch/ucair/18_047/blockface/block08/volumes/raw/difference_volume.mhd')
    # deformation = generate('18_047','block08',source_space='blockface',target_space='exvivo')
    # def_block = so.ApplyGrid.Create(deformation)(block, deformation)

    # target = io.LoadITKFile(
    #     '/home/sci/blakez/ucair/18_047/rawVolumes/PostImaging_2018-07-02/011_----_Day3_lr_NPV_Segmentation_047.nrrd')
    # ablation_vol = io.LoadITKFile('/home/sci/blakez/all_ablation_segs.nrrd')
    # target_resamp = so.ResampleWorld.Create(ablation_vol)(target)
    # numerator = 2 * (target_resamp.data * ablation_vol.data)
    # denomiator = (target_resamp.data + ablation_vol.data)
    # dice = (numerator.sum() + 1) / (denomiator.sum() + 1)

    # block_list = sorted(glob.glob(f'/hdscratch/ucair/{rabbit}/blockface/block*'))
    # for block_path in block_list:
    #     block = block_path.split('/')[-1]
    #     with open(f'{block_path}/surfaces/raw/{block}_deformable_config.yaml', 'r') as f:
    #         params = yaml.load(f, Loader=yaml.FullLoader)
    #     params['converge'] = 0.75
    #     params['currents_sigma'] = [3.0, 1.5]
    #     params['deformable_lr'] = [0.0002, 0.0001]
    #     if block == 'block04':
    #         params['grid_size'] = [20, 90, 90]
    #     else:
    #         params['grid_size'] = [20, 100, 100]
    #     params['propagation_sigma'] = [2.0, 2.0, 3.0]
    #     params['niter'] = 500
    #     with open(f'{block_path}/surfaces/raw/{block}_deformable_config.yaml', 'w') as f:
    #         yaml.dump(params, f)

    for block_path in block_list[:-2]:
        block = block_path.split('/')[-1]
        # generate_segmentation_volume(rabbit, block, base_dir=base_dir)
        # generate_image_volume(rabbit, block, base_dir=base_dir)
        # deform_histology_volumes(rabbit, block, base_dir=base_dir)
        # deform_image_volumes(rabbit, block, base_dir=base_dir)

    stack_image_volumes(rabbit, block_list, base_dir=base_dir)
    # stack_hitology_volumes(rabbit, block_list, base_dir=base_dir)
