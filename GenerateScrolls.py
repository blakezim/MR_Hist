import os
import sys
sys.path.append("/home/sci/blakez/code/")
import glob
import h5py
import torch
import numpy as np
from skimage import measure
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes, binary_opening, binary_closing

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridOperators as so

import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def generate_slice(slice_num, image_slice, segmentation_slice, out_path, aspect=1.0, color='gray', extra=None):

    contour_width = 1.8
    # Start with the ablated region if there is one
    part_contours = measure.find_contours(segmentation_slice.data.squeeze().cpu().numpy(), 0.5)
    if extra is not None:
        extra_contours = measure.find_contours(extra.data.squeeze().cpu().numpy(), 0.5)


    # Plot the original image
    plt.figure()
    plt.imshow(image_slice, cmap='gray', aspect=1.0/aspect, interpolation='nearest')
    plt.axis('off')
    plt.show()
    # plt.gca().invert_yaxis()
    # plt.savefig(f'{out_path}/Images/{slice_num:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)
    # plt.gca().invert_yaxis()

    if extra is not None:

        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0.0, 0.0, 0.0), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]
        cm = LinearSegmentedColormap.from_list('hist_color', colors, N=1)
        masked = np.ma.masked_where(extra.squeeze() == 0, extra.squeeze())
        plt.imshow(masked, cmap=cm, aspect=1.0 / aspect, alpha=0.7)
        # plt.axis('off')
        # plt.gca().invert_yaxis()
        # plt.gca().patch.set_facecolor([0, 0, 0, 0])

        # try:
        #     for contour in extra_contours:
        #         plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[6], linewidth=contour_width)
        # except IndexError:
        #     pass

    try:
        for contour in part_contours:
            plt.plot(contour[:, 1], contour[:, 0], color=color, linewidth=contour_width)
    except IndexError:
        pass

    plt.pause(1.0)

    plt.gca().invert_yaxis()
    plt.savefig(f'{out_path}/Shaded/im_{slice_num:03d}_shaded.png', dpi=600, bbox_inches='tight', pad_inches=0)

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

    tumor_seg_file = '/hdscratch/ucair/18_047/mri/exvivo/volumes/raw/T1_tumor_segmentation_exvivo_18_047.nrrd'
    ablation_seg_file = '/hdscratch/ucair/18_047/mri/exvivo/volumes/raw/T1_ablation_segmentation_exvivo_18_047.nrrd'
    exterior_seg_file = '/hdscratch/ucair/18_047/mri/exvivo/volumes/raw/T1_exterior_segmentation_18_047.nrrd'

    images = io.LoadITKFile(f'{ex_path}/011_----_3D_VIBE_0p5iso_cor_3ave.nrrd', device=device)
    tumor_seg = io.LoadITKFile(tumor_seg_file, device=device)
    ablation_seg = io.LoadITKFile(ablation_seg_file, device=device)
    exterior_seg = io.LoadITKFile(exterior_seg_file, device=device)

    tumor_seg = so.ResampleWorld.Create(images, device=device)(tumor_seg)
    ablation_seg = so.ResampleWorld.Create(images, device=device)(ablation_seg)
    exterior_seg = so.ResampleWorld.Create(images, device=device)(exterior_seg)

    tumor_seg.data = (tumor_seg.data >= 0.5).float()
    ablation_seg.data = (ablation_seg.data >= 0.5).float()
    exterior_seg.data = (exterior_seg.data >= 0.5).float()

    aspect = GetAspect(images)

    for s in range(100, 253):
        # Extract the slice
        im_slice = images[:, s].squeeze().cpu()
        seg_slice = tumor_seg[:, s].squeeze().cpu() + ablation_seg[:, s].squeeze().cpu()
        ext_slice = exterior_seg[:, s].squeeze().cpu()

        im_slice = im_slice[:, 80:-80]
        seg_slice = seg_slice[:, 80:-80]
        ext_slice = ext_slice[:, 80:-80]

        # plt.figure()
        # plt.imshow(im_slice, cmap='gray', aspect=1.0 / aspect, interpolation='nearest')
        # plt.axis('off')
        # plt.show()
        # plt.gca().invert_yaxis()
        # plt.savefig(f'{frame_dir}/Images/{s:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)
        # plt.gca().invert_yaxis()
        #
        # plt.close('all')

        generate_slice(s, im_slice, seg_slice, frame_dir, aspect, color=(0.114, 0.686, 0.666), extra=ext_slice)

        print(f'Done with slice {s}/{int(images.size[0])}')


def generate_invivo(ex_path, frame_dir):

    file = '011_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_POST_00.nii.gz'
    tumor_seg_file = '/hdscratch/ucair/18_047/mri/invivo/volumes/raw/Day3_tumor_segmentation_18_047.nrrd'
    ablation_seg_file = '/hdscratch/ucair/18_047/mri/invivo/volumes/raw/Day3_NPV_segmentation_18_047.nrrd'

    images = io.LoadITKFile(f'{ex_path}/{file}', device=device)
    tumor_seg = io.LoadITKFile(tumor_seg_file, device=device)
    ablation_seg = io.LoadITKFile(ablation_seg_file, device=device)

    tumor_seg = so.ResampleWorld.Create(images, device=device)(tumor_seg)
    ablation_seg = so.ResampleWorld.Create(images, device=device)(ablation_seg)

    tumor_seg.data = (tumor_seg.data >= 0.5).float()
    ablation_seg.data = (ablation_seg.data >= 0.5).float()

    aspect = GetAspect(images)

    for s in range(140, 201):
        # Extract the slice
        im_slice = images[:, s].squeeze().cpu()
        seg_slice = tumor_seg[:, s].squeeze().cpu() + ablation_seg[:, s].squeeze().cpu()

        im_slice = im_slice[:, 80:-80]
        seg_slice = seg_slice[:, 80:-80]

        plt.figure()
        plt.imshow(im_slice, cmap='gray', aspect=1.0 / aspect, interpolation='nearest')
        plt.axis('off')
        plt.show()
        plt.gca().invert_yaxis()
        plt.savefig(f'{frame_dir}/Images/{s:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)

        plt.close('all')

        generate_slice(s, im_slice, seg_slice, frame_dir, aspect, color=(0.8, 0.314, 0.016))

        print(f'Done with slice {s}/{int(images.size[0])}')


def generate_invivo_hist(ex_path, frame_dir):

    if not os.path.exists(frame_dir):
        os.makedirs(f'{frame_dir}/Images')
        os.makedirs(f'{frame_dir}/Contours')

    # file = '011_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_POST_00.nii.gz'
    # npv_seg_file = '/hdscratch/ucair/18_047/mri/invivo/volumes/raw/Day3_NPV_segmentation_18_047.nrrd'
    # hst_seg_file = '/hdscratch/ucair/18_047/microscopic/recons/all_ablation_segs_to_invivo.nrrd'

    # file = '028_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_Post_01.nii.gz'
    file = '/home/sci/blakez/ucair/longitudinal/18_062/Day0_contrast_VIBE.nii.gz'
    t1_nc_file = '/home/sci/blakez/ucair/longitudinal/18_062/Day0_non_contrast_VIBE.nii.gz'
    # npv_seg_file = '/hdscratch/ucair/18_062/mri/invivo/volumes/raw/028_----_Day3_lr_NPV_Segmentation_062.nrrd'
    npv_seg_file = '/home/sci/blakez/ucair/longitudinal/18_062/Day0_NPV_Segmentation_062.nrrd'
    hst_seg_file = '/hdscratch/ucair/18_062/microscopic/recons/all_ablation_segs_to_invivo.mhd'
    log_ctd = '/hdscratch/ucair/AcuteBiomarker/Data/18_062/18_062_log_ctd_map.nii.gz'
    ctd_mask = '/hdscratch/ucair/AcuteBiomarker/Data/18_062/18_062_tissue_seg.nii.gz'

    images = io.LoadITKFile(file, device=device)
    t1_nc = io.LoadITKFile(t1_nc_file, device=device)
    npv_seg = io.LoadITKFile(npv_seg_file, device=device)
    hst_seg = io.LoadITKFile(hst_seg_file, device=device)
    log_ctd = io.LoadITKFile(log_ctd, device=device)
    ctd_mask = io.LoadITKFile(ctd_mask, device=device)
    log_ctd = log_ctd * ctd_mask

    npv_seg = so.ResampleWorld.Create(images, device=device)(npv_seg)
    t1_nc = so.ResampleWorld.Create(images, device=device)(t1_nc)
    hst_seg = so.ResampleWorld.Create(images, device=device)(hst_seg)
    log_ctd = so.ResampleWorld.Create(images, device=device)(log_ctd)

    npv_seg.data = (npv_seg.data >= 0.5).float()
    hst_seg.data = (hst_seg.data >= 0.5).float()

    # hst_seg = torch.tensor(binary_erosion(binary_dilation(hst_seg.data.cpu().squeeze(), iterations=4), iterations=4))
    # hst_seg = hst_seg.unsqueeze(0)

    aspect = GetAspect(images)

    ### ADDED
    #
    # def create_circular_mask(h, w, center=None, radius=None):
    #
    #     if center is None:  # use the middle of the image
    #         center = (int(w / 2), int(h / 2))
    #     if radius is None:  # use the smallest distance between the center and image walls
    #         radius = min(center[0], center[1], w - center[0], h - center[1])
    #
    #     Y, X = np.ogrid[:h, :w]
    #     dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    #
    #     mask = dist_from_center <= radius
    #     return mask
    #
    # slice_n = 130
    # green = matplotlib._cm._tab10_data[2]
    # blue = matplotlib._cm._tab10_data[0]
    # out_dir = '/home/sci/blakez/papers/dissertation/Day0and3NPV/'

    try:
        deform = io.LoadITKFile(f'/hdscratch/ucair/AcuteBiomarker/Data/{rabbit}/{rabbit}_day3_to_day0_phi_inv.nii.gz', device=device)
    except:
        def_file = f'/home/sci/blakez/ucair/longitudinal/{rabbit}/'
        def_file += 'deformation_fields/Day3_non_contrast_VIBE_interday_deformation_incomp.nii.gz'
        deform = io.LoadITKFile(def_file, device=device)

    def_hist = so.ApplyGrid.Create(deform, device=device)(hst_seg, deform)
    def_hist.to_('cpu')
    hst_seg.data = (hst_seg.data >= 0.5).float()

    def_hist_np = binary_erosion(binary_dilation(def_hist.data.squeeze(), iterations=4), iterations=4)
    def_hist.data = torch.tensor(def_hist_np).unsqueeze(0).float()
    hst_seg = def_hist
    # plt.figure()
    # plt.imshow(t1_nc.data.cpu()[0, slice_n].squeeze(), aspect=1.0 / aspect, cmap='gray')
    # plt.gca().invert_yaxis()
    # plt.axis('off')
    # plt.savefig(f'{out_dir}/day0_nc_t1.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    #
    # plt.figure()
    # plt.imshow(images.data.cpu()[0, slice_n].squeeze(), aspect=1.0 / aspect, cmap='gray')
    # plt.gca().invert_yaxis()
    # plt.axis('off')
    # plt.savefig(f'{out_dir}/day0_c_t1.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

    # slice_im = t1_nc.data.cpu()[0, slice_n].squeeze()
    # zero_slice = np.zeros_like(slice_im)
    # masked_slice = np.ma.masked_where(zero_slice == 0.0, zero_slice).squeeze()
    # plt.figure()
    # plt.imshow(masked_slice.squeeze(), aspect=1.0 / aspect, cmap='gray')
    # npv_slice = hst_seg.data.cpu()[0, slice_n].squeeze()
    # npv_contours = measure.find_contours(npv_slice.data.squeeze().cpu().numpy(), 0.5)
    # for contour in npv_contours:
    #     plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[6], linewidth=1.8)
    # plt.gca().invert_yaxis()
    # plt.gca().patch.set_facecolor([0, 0, 0, 0])
    # plt.axis('off')
    # plt.savefig(f'{out_dir}/day0_contour_hist.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    #
    # from matplotlib.colors import LinearSegmentedColormap
    # colors = [(0.0, 0.0, 0.0), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]
    # cm = LinearSegmentedColormap.from_list('hist_color', colors, N=1)
    # plt.figure()
    # masked = np.ma.masked_where(hst_seg.data.cpu()[0, slice_n].squeeze() == 0, hst_seg.data.cpu()[0, slice_n].squeeze())
    # plt.imshow(masked, cmap=cm,aspect=1.0 / aspect, alpha=0.5)
    # plt.axis('off')
    # plt.gca().invert_yaxis()
    # plt.gca().patch.set_facecolor([0, 0, 0, 0])
    # plt.savefig(f'{out_dir}/day3_shaded_hist.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    # #
    # circle_mask = create_circular_mask(112, 533, center=[280, 40], radius=50)
    # outside_FOV = log_ctd.data[0, slice_n].cpu().numpy() > 0
    # ctd_mask = np.logical_and(outside_FOV, circle_mask == 1)
    # ctd_slice = torch.exp(log_ctd.data[0, slice_n].cpu())
    # ctd_mask = np.logical_and(ctd_slice > 10.0, circle_mask == 1)
    # masked_thermal = np.ma.masked_where(ctd_mask == 0.0, ctd_slice.numpy()).squeeze()
    # plt.figure()
    # # plt.imshow(t1_nc.data.cpu()[0, slice_n].squeeze(), aspect=1.0 / aspect, cmap='gray')
    # plt.imshow(masked_thermal, aspect=1.0 / aspect, cmap='jet', vmin=0.5, vmax=240)
    # plt.gca().patch.set_facecolor([0, 0, 0, 0])
    # plt.gca().invert_yaxis()
    # plt.axis('off')
    # # plt.savefig(f'{out_dir}/day0_ctd_no_cmap.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.colorbar()
    # # plt.savefig(f'{out_dir}/day0_ctd_cmap.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    # slice_im = t1_nc.data.cpu()[0, slice_n].squeeze()
    # zero_slice = np.zeros_like(slice_im)
    # masked_slice = np.ma.masked_where(zero_slice == 0.0, zero_slice).squeeze()
    # plt.figure()
    # plt.imshow(masked_slice.squeeze(), aspect=1.0 / aspect, cmap='gray')
    # npv_slice = log_ctd.data.cpu()[0, slice_n].squeeze()
    # npv_contours = measure.find_contours(npv_slice.data.squeeze().cpu().numpy() * circle_mask, np.log(240))
    # for contour in npv_contours:
    #     plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1.8)
    # plt.axis('off')
    # plt.gca().invert_yaxis()
    # plt.gca().patch.set_facecolor([0, 0, 0, 0])
    # plt.savefig(f'{out_dir}/day0_ctd_contour.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    # circle_mask = create_circular_mask(112, 533, center=[280, 40], radius=50)
    # for s in range(140, 201):
    for s in range(70, 150):
        # for s in range(100, 180):
        # Extract the slice
        im_slice = t1_nc[:, s].squeeze().cpu()
        # seg_slice = npv_seg[:, s].squeeze().cpu()
        seg_slice = log_ctd[:, s].squeeze().cpu()
        ext_slice = hst_seg[:, s].squeeze().cpu()
        #

        # ctd_mask = np.logical_and(seg_slice > 2.3, circle_mask == 1)
        # seg_slice = seg_slice * ctd_mask
        seg_slice = (seg_slice >= np.log(240)).float()
        # im_slice = im_slice[:, 80:-80]
        # seg_slice = seg_slice[:, 80:-80]
        # ext_slice = ext_slice[:, 80:-80]

        # plt.figure()
        # plt.imshow(im_slice, cmap='gray', aspect=1.0 / aspect, interpolation='nearest')
        # plt.axis('off')
        # plt.show()
        # plt.gca().invert_yaxis()
        # plt.savefig(f'{frame_dir}/Images/{s:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)
        #
        # plt.close('all')

        generate_slice(s, im_slice, seg_slice, frame_dir, aspect, color='red', extra=ext_slice)
        # generate_slice(s, im_slice, seg_slice, frame_dir, aspect, color='red', extra=ext_slice)

        print(f'Done with slice {s}/{int(images.size[0])}')


def generate_day0(ex_path, frame_dir, base_dir='/hdscratch/ucair/'):

    if not os.path.exists(frame_dir):
        os.makedirs(f'{frame_dir}/Images')
        os.makedirs(f'{frame_dir}/Contours')

    # file = '011_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_POST_00.nii.gz'
    # npv_seg_file = '/hdscratch/ucair/18_047/mri/invivo/volumes/raw/Day3_NPV_segmentation_18_047.nrrd'
    # hst_seg_file = '/hdscratch/ucair/18_047/microscopic/recons/all_ablation_segs_to_invivo.nrrd'

    file = '/home/sci/blakez/ucair/longitudinal/18_047/Day0_contrast_VIBE.nii.gz'
    # file = '028_----_3D_VIBE_1mmIso_NoGrappa_1avg_fatsat_cor_Post_01.nii.gz'
    npv_seg_file = '/home/sci/blakez/ucair/longitudinal/18_047/Day0_Tumor_Segmentation_047.nrrd'
    # hst_seg_file = '/hdscratch/ucair/18_062/microscopic/recons/all_ablation_segs_to_invivo.mhd'

    images = io.LoadITKFile(file, device=device)
    npv_seg = io.LoadITKFile(npv_seg_file, device=device)
    # hst_seg = io.LoadITKFile(hst_seg_file, device=device)

    npv_seg = so.ResampleWorld.Create(images, device=device)(npv_seg)
    # hst_seg = so.ResampleWorld.Create(images, device=device)(hst_seg)

    npv_seg.data = (npv_seg.data >= 0.5).float()
    # hst_seg.data = (hst_seg.data >= 0.5).float()

    # hst_seg = torch.tensor(binary_erosion(binary_dilation(hst_seg.data.cpu().squeeze(), iterations=4), iterations=4))
    # hst_seg = hst_seg.unsqueeze(0)

    aspect = GetAspect(images)

    # for s in range(140, 201):
    for s in range(30, 85):
        # Extract the slice
        im_slice = images[:, s].squeeze().cpu()
        seg_slice = npv_seg[:, s].squeeze().cpu()
        # ext_slice = hst_seg[:, s].squeeze().cpu()

        # im_slice = im_slice[:, 80:-80]
        # seg_slice = seg_slice[:, 80:-80]
        # ext_slice = ext_slice[:, 80:-80]

        # plt.figure()
        # plt.imshow(im_slice, cmap='gray', aspect=1.0 / aspect, interpolation='nearest')
        # plt.axis('off')
        # plt.show()
        # plt.gca().invert_yaxis()
        # plt.savefig(f'{frame_dir}/Images/{s:03d}_image.png', dpi=600, bbox_inches='tight', pad_inches=0)
        #
        # plt.close('all')

        generate_slice(s, im_slice, seg_slice, frame_dir, aspect, color='gold', extra=None)

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
    rabbit = '18_062'
    base_dir = '/hdscratch2/'
    # ex_dir = f'/hdscratch/rabbit_data/{rabbit}/rawVolumes/ExVivo_2018-07-26/'
    # out_dir = f'/home/sci/blakez/ucair/Animations/Scrolls/Exvivo_T1/'
    # generate_exvivo(ex_dir, out_dir)

    # in_dir = f'/hdscratch/rabbit_data/{rabbit}/rawVolumes/PostImaging_2018-07-02/'
    in_dir = f'/scratch/rabbit_data/{rabbit}/rawVolumes/Ablation_2018-06-28/'
    out_dir = '/home/sci/blakez/ucair/Animations/Scrolls/Day0_CTD_Hist_Scroll/'
    # generate_invivo(in_dir, out_dir)
    generate_invivo_hist(in_dir, out_dir)
    # generate_day0(in_dir, out_dir, base_dir)

    # in_dir = '/hdscratch/ucair/blockface/18_047/affineImages/block07/surface/'
    # out_dir = '/home/sci/blakez/ucair/Animations/Scrolls/Blockface/'
    # generate_blockface(in_dir, out_dir)
