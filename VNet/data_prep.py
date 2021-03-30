import os
import glob
import torch
import numpy as np
import SimpleITK as sitk

# import matplotlib
# matplotlib.use('qt5agg')
# import matplotlib.pyplot as plt
# plt.ion()


def preprocess_mhd_image(image_path):
    itk_image = sitk.ReadImage(image_path)
    numpy_image = np.squeeze(sitk.GetArrayFromImage(itk_image))
    return torch.from_numpy(numpy_image)


def preprocess_nrrd_image(image_path):
    itk_image = sitk.ReadImage(image_path)
    numpy_image = np.squeeze(sitk.GetArrayFromImage(itk_image))
    return torch.from_numpy(numpy_image)


def preprocess_tif_image(image_path):
    itk_image = sitk.ReadImage(image_path)
    numpy_image = np.squeeze(sitk.GetArrayFromImage(itk_image))
    return torch.from_numpy(numpy_image)


def generate_input_block(rabbit_path, block, opt=None, shape=None):

    # hd_root = f'/hdscratch/ucair/{rabbit}/blockface/{block}/volumes/raw/'

    # Get the image lists for the block
    # surface_pwds = sorted(glob.glob(f'{hd_root}/surface/*.mhd'))
    # scatter_pwds = sorted(glob.glob(f'{hd_root}/scatter/*.mhd'))
    #
    # surf_vol = []
    # scat_vol = []
    #
    # for surf, scat in zip(surface_pwds, scatter_pwds):
    #     # Define the path to the output folder
    #     surf_vol.append(preprocess_mhd_image(surf).unsqueeze(0))
    #     scat_vol.append(preprocess_mhd_image(scat).unsqueeze(0))

    surf_vol = preprocess_mhd_image(f'{rabbit_path}/blockface/{block}/volumes/raw/surface_volume.mhd')
    scat_vol = preprocess_mhd_image(f'{rabbit_path}/blockface/{block}/volumes/raw/scatter_volume.mhd')
    surf_vol = surf_vol.permute(3, 0, 1, 2)
    scat_vol = scat_vol.permute(3, 0, 1, 2)

    orig_shape = surf_vol.shape

    # If the shape is None, resample it to be multiples of opt.block
    if not shape:
        shape = [
            64,
            orig_shape[2] // 6 + (opt.cube[1] - (orig_shape[2] // 6 % opt.cube[1])),
            orig_shape[3] // 6 + (opt.cube[2] - (orig_shape[3] // 6 % opt.cube[2]))
        ]
    # Resample the images to the new size
    surf_vol = torch.nn.functional.interpolate(surf_vol.unsqueeze(0), size=shape,
                                               mode='trilinear', align_corners=True).squeeze()
    scat_vol = torch.nn.functional.interpolate(scat_vol.unsqueeze(0), size=shape,
                                               mode='trilinear', align_corners=True).squeeze()
    diff_vol = surf_vol - scat_vol

    grad_scat = scat_vol[:, :-1, :, :].cpu() - scat_vol[:, 1:, :, :].cpu()
    grad_scat = (grad_scat - grad_scat.min()) / (grad_scat.max() - grad_scat.min())
    grad_scat = torch.cat([grad_scat[:, 0, :, :].unsqueeze(1), grad_scat], dim=1)

    grad_surf = surf_vol[:, :-1, :, :].cpu() - surf_vol[:, 1:, :, :].cpu()
    grad_surf = (grad_surf - grad_surf.min()) / (grad_surf.max() - grad_surf.min())
    grad_surf = torch.cat([grad_surf[:, 0, :, :].unsqueeze(1), grad_surf], dim=1)

    grad_diff = diff_vol[:, :-1, :, :].cpu() - diff_vol[:, 1:, :, :].cpu()
    grad_diff = (grad_diff - grad_diff.min()) / (grad_diff.max() - grad_diff.min())
    grad_diff = torch.cat([grad_diff[:, 0, :, :].unsqueeze(1), grad_diff], dim=1)

    # Concatenate the surface and scatter now
    input_vol = torch.cat([surf_vol, scat_vol, diff_vol, grad_surf, grad_scat, grad_diff])

    return input_vol, orig_shape


def complile_training_data(rabbits):

    hd_root = '/hdscratch/ucair/'
    sd_root = '/usr/sci/scratch/blakez/blockface_data/'

    for rabbit in rabbits:

        # Get the list of blocks that have segmentations
        rabbit_path = f'{hd_root}{rabbit}/'
        block_pwds = sorted(glob.glob(f'{rabbit_path}/blockface/block*'))
        if not block_pwds:
            rabbit_path = f'/usr/sci/scratch/blakez/rabbit_data/{rabbit}/'
            block_pwds = sorted(glob.glob(f'{rabbit_path}/blockface/block*'))

        blocks = [x.split('/')[-1] for x in block_pwds]

        for block in blocks:
            print(f'===> Generating {rabbit}/{block} ... ', end='')
            try:
                label_vol = preprocess_nrrd_image(
                    f'{rabbit_path}blockface/{block}/volumes/raw/segmentation_volume.nrrd')
            except RuntimeError:
                label_vol = preprocess_mhd_image(
                    f'{rabbit_path}blockface/{block}/volumes/raw/segmentation_volume.mhd')

            new_size = [64, label_vol.shape[1] // 4, label_vol.shape[2] // 4]
            label_vol = torch.nn.functional.interpolate(label_vol.unsqueeze(0).unsqueeze(0).float(), size=new_size,
                                                        mode='trilinear', align_corners=True).squeeze()

            input_vol, _ = generate_input_block(rabbit_path, block, opt=None, shape=new_size)

            if not os.path.exists(f'{sd_root}inputs/'):
                os.makedirs(f'{sd_root}inputs/')

            if not os.path.exists(f'{sd_root}labels/'):
                os.makedirs(f'{sd_root}labels/')

            torch.save(input_vol, f'{sd_root}inputs/{rabbit}_{block}.pth')
            torch.save(label_vol, f'{sd_root}labels/{rabbit}_{block}.pth')

            print('Done')


if __name__ == '__main__':
    complile_training_data(['18_047', '18_060', '18_062'])
