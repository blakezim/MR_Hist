import os
import sys
sys.path.append("/home/sci/blakez/code/")
import glob
import h5py
import torch
import numpy as np
from skimage import measure

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridOperators as so

import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def generate_slice(slice_num, image_slice, segmentation_slice, out_path, aspect=1.0, color='gray'):

    contour_width = 0.8

    # Start with the ablated region if there is one
    # part_contours = measure.find_contours(segmentation_slice.data.squeeze().cpu().numpy(), 0.5)

    # zeros = np.zeros_like(image_slice)


    # Plot the original image
    plt.figure()
    plt.imshow(image_slice, cmap='gray', aspect=1.0/aspect, interpolation='nearest')
    plt.axis('off')
    plt.show()
    plt.gca().invert_yaxis()
    plt.savefig(f'{out_path}/Images/{slice_num:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.gca().invert_yaxis()

    # try:
    #     for contour in part_contours:
    #         plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=contour_width)
    # except IndexError:
    #     pass

    # plt.pause(1.0)
    #
    # plt.gca().invert_yaxis()
    # plt.savefig(f'{out_path}/Contours/{slice_num:03d}_contoured.png', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.close('all')


def GetAspect(Image, axis='default', retFloat=True):
    '''given a image grid, determines the 2D aspect ratio of the
    off-dimension'''

    imsz = Image.size.tolist()

    # Might need to flip these two
    aspect = (Image.spacing[0] / Image.spacing[1]).item()
    sz = [imsz[1], imsz[0]]

    if axis == 'cart':
        aspect = 1.0/aspect
        sz = sz[::-1]

    if retFloat:
        return aspect

    if aspect > 1:
        scale = [sz[0]/aspect, sz[1]*1.0]
    else:
        scale = [sz[0]*1.0, sz[1]*aspect]

    # scale incorporates image size (grow if necessary) and aspect ratio
    while scale[0] <= 400 and scale[1] <= 400:
        scale = [scale[0]*2, scale[1]*2]

    return [int(round(scale[0])), int(round(scale[1]))]


def generate_exvivo(ex_path, frame_dir):
    images = io.LoadITKFile(f'{ex_path}/011_----_3D_VIBE_0p5iso_cor_3ave.nrrd', device=device)

    tumor_seg = so.ResampleWorld.Create(images, device=device)(tumor_seg)

    tumor_seg.data[tumor_seg.data < 0.5] = 0.0
    tumor_seg.data[tumor_seg.data >= 0.5] = 1.0

    aspect = GetAspect(images)

    for s in range(100, 253):
        # Extract the slice
        im_slice = images[:, s].squeeze().cpu()
        # seg_slice = tumor_seg[:, s].squeeze().cpu() + ablation_seg[:, s].squeeze().cpu()

        im_slice = im_slice[:, 80:-80]

        plt.figure()
        plt.imshow(im_slice, cmap='gray', aspect=1.0 / aspect, interpolation='nearest')
        plt.axis('off')
        plt.show()
        plt.gca().invert_yaxis()
        plt.savefig(f'{frame_dir}/Images/{s:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)
        plt.gca().invert_yaxis()

        plt.close('all')

        # generate_slice(s, im_slice, seg_slice, frame_dir, aspect, color=(0.114, 0.686, 0.666))

        print(f'Done with slice {s}/{int(images.size[0])}')


def generate_invivo(ex_path, frame_dir):

    file = '011_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_POST_00.nii.gz'

    images = io.LoadITKFile(f'{ex_path}/{file}', device=device)

    aspect = GetAspect(images)

    for s in range(190, 201):
        # Extract the slice
        im_slice = images[:, s].squeeze().cpu()
        # seg_slice = tumor_seg[:, s].squeeze().cpu() + ablation_seg[:, s].squeeze().cpu()

        im_slice = im_slice[:, 80:-80]

        plt.figure()
        plt.imshow(im_slice, cmap='gray', aspect=1.0 / aspect, interpolation='nearest')
        plt.axis('off')
        plt.show()
        plt.gca().invert_yaxis()
        plt.savefig(f'{frame_dir}/Images/{s:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)

        plt.close('all')

        # generate_slice(s, im_slice, seg_slice, frame_dir, aspect, color=(0.114, 0.686, 0.666))

        print(f'Done with slice {s}/{int(images.size[0])}')


def generate_blockface(in_dir, frame_dir):
    list = sorted(glob.glob(f'{in_dir}/*.mhd'))

    for i, file in enumerate(list):
        if i==2 or i == 39:

            image = io.LoadITKFile(f'{file}').data.squeeze().permute(1, 2, 0)

            plt.figure()
            plt.imshow(image, cmap='gray', aspect=1.0, interpolation='nearest')
            plt.axis('off')
            plt.show()
            plt.gca().invert_yaxis()
            plt.savefig(f'{frame_dir}/Images/{i:03d}_image.png', dpi=500, bbox_inches='tight', pad_inches=0)

            plt.close('all')

            print(f'Done with slice {i+1}/{len(list)}')

if __name__ == '__main__':
    rabbit = '18_047'
    # ex_dir = f'/home/sci/blakez/ucair/{rabbit}/rawVolumes/ExVivo_2018-07-26/'
    # out_dir = f'/home/sci/blakez/ucair/Animations/Scrolls/Exvivo_T1/'
    # generate_exvivo(ex_dir, out_dir)

    # in_dir = f'/home/sci/blakez/ucair/{rabbit}/rawVolumes/PostImaging_2018-07-02/'
    # out_dir = '/home/sci/blakez/ucair/Animations/Scrolls/Invivo_CET1/'
    # generate_invivo(in_dir, out_dir)

    in_dir = '/hdscratch/ucair/blockface/18_047/affineImages/block07/surface/'
    out_dir = '/home/sci/blakez/ucair/Animations/Scrolls/Blockface/'
    generate_blockface(in_dir, out_dir)