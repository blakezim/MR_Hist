import os
import sys
import glob
sys.path.append("/home/sci/blakez/code/")

import h5py
import torch
import numpy as np
from skimage import measure

import CAMP.camp.Core as core
import CAMP.camp.FileIO as io
import CAMP.camp.StructuredGridOperators as so

import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def generate_figures(slice, segs, out_path, base_name, save=True, extra_cont=None):

    contour_width = 1.8
    shape = slice.data.permute(1, 2, 0).shape[0:2]
    one_mm = 1.0 / slice.spacing[0]
    scale = 2.0
    scale_pix = scale * one_mm
    x_off = 200 # 500
    y_off = 600

    part_contours = []

    # Start with the ablated region if there is one
    if len(segs) == 3:
        accu_seg = segs[-1].copy()
        part_contours += [measure.find_contours(accu_seg.data.squeeze().cpu().numpy(), 0.5)]
        accu_seg = accu_seg + segs[-2]
        mask = (1 - accu_seg.data).squeeze().cpu().numpy()
        part_contours += [measure.find_contours(accu_seg.data.squeeze().cpu().numpy(), 0.5, mask=mask.astype(bool))]
        accu_seg = accu_seg + segs[-3]
        mask = (1 - accu_seg.data).squeeze().cpu().numpy()
        part_contours += [measure.find_contours(accu_seg.data.squeeze().cpu().numpy(), 0.5, mask=mask.astype(bool))]

    elif len(segs) == 2:
        accu_seg = segs[-1].copy()
        part_contours += [measure.find_contours(accu_seg.data.squeeze().cpu().numpy(), 0.5)]
        accu_seg = accu_seg + segs[-2]
        mask = (1 - accu_seg.data).squeeze().cpu().numpy()
        part_contours += [measure.find_contours(accu_seg.data.squeeze().cpu().numpy(), 0.5, mask=mask.astype(bool))]

    else:
        accu_seg = segs[-1].copy()
        part_contours += [measure.find_contours(accu_seg.data.squeeze().cpu().numpy(), 0.5)]

    part_contours = part_contours[::-1]

    #
    # plt.gca().patch.set_facecolor([0, 0, 0, 0])
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(test_alpha)
    # plt.savefig('/home/sci/blakez/test_transparent.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

    # Plot the original image
    masked_slice = np.ma.masked_where(slice.data.cpu() == 0, slice.data.cpu()).squeeze()

    # Shade the region of the ablated histology
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0.0, 0.0, 0.0), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]
    cm = LinearSegmentedColormap.from_list('hist_color', colors, N=1)
    plt.figure()
    masked = np.ma.masked_where(segs[-1].data.cpu().squeeze() == 0, segs[-1].data.cpu().squeeze())
    plt.imshow(masked, cmap=cm, alpha=0.5)
    plt.axis('off')
    if save:
        plt.savefig(f'{out_path}/{base_name}_shaded_ablated_hist.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    #

    if masked_slice.shape[0] == 3:
        slice_np = slice.data.cpu().permute(1, 2, 0)
        masked_slice = torch.cat([slice_np, (slice_np[:, :, 0] > 0).float().unsqueeze(-1)], -1)
    else:
        masked_slice = masked_slice.squeeze()
    plt.figure()
    plt.imshow(masked_slice, cmap='gray')
    plt.plot([y_off, y_off + scale_pix], [shape[0] - x_off, shape[0] - x_off], 'k-', linewidth=contour_width)
    plt.axis('off')
    plt.gca().patch.set_facecolor([0, 0, 0, 0])
    plt.show()
    if save:
        plt.savefig(f'{out_path}/{base_name}_original.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

    # Plot the image with contours
    plt.figure()
    plt.imshow(masked_slice, cmap='gray')
    plt.plot([y_off, y_off + scale_pix], [shape[0] - x_off, shape[0] - x_off], 'k-', linewidth=contour_width)

    # for contour in part_contours[0]:
    #     plt.plot(contour[:, 1], contour[:, 0], color='crimson', linewidth=contour_width)

    if extra_cont:
        for contour in extra_cont:
            plt.plot(contour[:, 1], contour[:, 0], 'blue', linewidth=contour_width)

    # try:
    #     for contour in part_contours[1]:
    #         plt.plot(contour[:, 1], contour[:, 0], 'lime', linewidth=contour_width)
    # except IndexError:
    #     pass

    try:
        for contour in part_contours[2]:
            plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[6], linewidth=contour_width)
    except IndexError:
        pass

    plt.axis('off')
    plt.gca().patch.set_facecolor([0, 0, 0, 0])
    plt.show()
    plt.pause(1.0)
    if save:
        plt.savefig(f'{out_path}/{base_name}_with_contours.png', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.close('all')


def get_mr_slice(mr_file, blockface, slice_num, sample_device='cpu'):

    orig_device = blockface.device

    mr_data = io.LoadITKFile(mr_file, device=sample_device)
    blockface.to_(sample_device)
    mr_resamp = so.ResampleWorld.Create(blockface, device=sample_device)(mr_data)
    mr_slice = mr_resamp.extract_slice(slice_num - 1, dim=0)

    mr_slice.to_(orig_device)
    blockface.to_(orig_device)

    del mr_data, mr_resamp
    torch.cuda.empty_cache()

    return mr_slice


def sample_on_histopathology(rabbit, block, img_num, bf_slice):

    blockface_dir = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'
    histology_dir = f'/hdscratch/ucair/{rabbit}/microscopic/{block}/'
    invivo_dir = f'/hdscratch/ucair/{rabbit}/mri/invivo/volumes/deformable/{block}/'
    day0_dir = f'/hdscratch/ucair/{rabbit}/mri/day0/volumes/deformable/{block}/'
    invivo_mr_out_path = f'{invivo_dir}/IMG_{img_num}/'
    day0_mr_out_path = f'{day0_dir}/IMG_{img_num}/'

    mic_file = f'{histology_dir}hdf5/{block}_img{img_num}_image.hdf5'

    # First need to see if the deformation from histology to blockface exists
    # Load the deformation
    try:
        phi_inv_data = io.LoadITKFile(
            f'{histology_dir}deformations/img_{img_num}_deformation_to_blockface.mhd', device=device
        )
        aff = np.loadtxt(f'{histology_dir}deformations/img_{img_num}_affine_to_blockface.txt')
        aff = torch.tensor(aff, device=device, dtype=torch.float32)
    except IOError:
        raise IOError(f'The full deformation for IMG {img_num} was not found. Please generate and then re-run.')

    # Because I can't save 2D deformations at the moment
    phi_inv = core.StructuredGrid(
        size=phi_inv_data.size[0:2],
        spacing=phi_inv_data.spacing[1:3],
        origin=phi_inv_data.origin[1:3],
        device=phi_inv_data.device,
        tensor=phi_inv_data.data.squeeze().permute(2, 0, 1),
        channels=2
    )

    # Apply the inverse affine to the deformation
    aff = aff.inverse()
    a = aff[0:2, 0:2].float()
    t = aff[-0:2, 2].float()

    phi_inv.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                phi_inv.data.permute(list(range(1, 2 + 1)) + [0]).unsqueeze(-1))
    phi_inv.data = (phi_inv.data.squeeze() + t).permute([-1] + list(range(0, 2)))

    if not os.path.exists(invivo_mr_out_path):
        os.makedirs(invivo_mr_out_path)

    if not os.path.exists(day0_mr_out_path):
        os.makedirs(day0_mr_out_path)

    meta_dict = {}
    with h5py.File(mic_file, 'r') as f:
        mic = f['RawImage/ImageData'][:, ::10, ::10]
        for key in f['RawImage'].attrs:
            meta_dict[key] = f['RawImage'].attrs[key]

    mic = core.StructuredGrid(
        mic.shape[1:],
        tensor=torch.tensor(mic, dtype=torch.float32, device=device),
        spacing=torch.tensor([10.0, 10.0], dtype=torch.float32, device=device),
        origin=(torch.tensor(mic.shape[1:]) / 2) * -1,
        device=device,
        dtype=torch.float32,
        channels=3
    )

    mic = (mic - mic.min()) / (mic.max() - mic.min())

    segs = []
    segs += [io.LoadITKFile(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_healthy_tissue.nrrd',
                            device=device)]
    if os.path.exists(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_uncertain_region.nrrd'):
        segs += [io.LoadITKFile(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_uncertain_region.nrrd',
                                device=device)]
    if os.path.exists(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_ablated_region.nrrd'):
        segs += [io.LoadITKFile(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_ablated_region.nrrd',
                                device=device)]

    # Apply the deformation to the microscopic image
    deformed_histology = so.ApplyGrid(phi_inv, device=device)(mic, phi_inv)
    for i, vol in enumerate(segs):
        segs[i] = so.ApplyGrid(phi_inv, device=device)(vol, phi_inv)

    # Load the blockface
    blockface = io.LoadITKFile(f'{blockface_dir}volumes/raw/difference_volume.mhd', device=device)

    sd = 'cuda:0'

    mr_invivo_t1_slice = get_mr_slice(f'{invivo_dir}/invivo_ce_t1_to_{block}.mhd', blockface, bf_slice - 1, sd)
    # mr_invivo_t2_slice = get_mr_slice(f'{invivo_dir}/invivo_t2_to_{block}.mhd', blockface, bf_slice - 1, sd)
    # mr_invivo_adc_slice = get_mr_slice(f'{invivo_dir}/invivo_adc_to_{block}.mhd', blockface, bf_slice - 1, sd)
    mr_invivo_npv_slice = get_mr_slice(f'{invivo_dir}/invivo_npv_to_{block}.mhd', blockface, bf_slice - 1, sd)
    mr_d0_npv_slice = get_mr_slice(f'/hdscratch/ucair/18_062/mri/day0/volumes/deformable/block04/day0_npv_to_{block}.mhd', blockface, bf_slice - 1, sd)
    mr_d0_t1_slice = get_mr_slice(
        f'/hdscratch/ucair/18_062/mri/day0/volumes/deformable/block04/day0_t1_to_{block}.mhd', blockface, bf_slice - 1,
        sd)
    mr_d0_t1_nc_slice = get_mr_slice(
        f'/hdscratch/ucair/18_062/mri/day0/volumes/deformable/block04/day0_t1_nc_to_{block}.mhd', blockface, bf_slice - 1,
        sd)
    mr_log_ctd_slice = get_mr_slice(
        f'/hdscratch/ucair/18_062/mri/day0/volumes/deformable/block04/day0_log_ctd_to_{block}.mhd', blockface, bf_slice - 1,
        sd)
    mr_log_ctd_slice.data = torch.exp(mr_log_ctd_slice.data)

    def create_circular_mask(h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    circle_mask = create_circular_mask(2586, 3426, center=[1831, 1100], radius=700)
    mr_log_ctd_slice.data = mr_log_ctd_slice.data * torch.tensor(circle_mask, device=device).float()
    # mr_day0_t2_slice = get_mr_slice(f'{day0_dir}/day0_t2_to_{block}.mhd', blockface, bf_slice - 1, sd)
    # mr_day0_ctd_slice = get_mr_slice(f'{day0_dir}/day0_ctd_to_{block}.mhd', blockface, bf_slice - 1, sd)
    # mr_day0_t1_slice = get_mr_slice(f'{day0_dir}/day0_ce_t1_to_{block}.mhd', blockface, bf_slice - 1, sd)
    # mr_day0_npv_slice = get_mr_slice(f'{day0_dir}/day0_npv_to_{block}.mhd', blockface, bf_slice - 1, sd)

    # io.SaveITKFile(mr_invivo_t1_slice, f'{invivo_mr_out_path}/invivo_ce_t1_as_bf_img_{img_num}.mhd')
    # io.SaveITKFile(mr_invivo_t2_slice, f'{invivo_mr_out_path}/invivo_t2_as_bf_img_{img_num}.mhd')
    # io.SaveITKFile(mr_invivo_adc_slice, f'{invivo_mr_out_path}/invivo_adc_as_bf_img_{img_num}.mhd')
    # io.SaveITKFile(mr_invivo_npv_slice, f'{invivo_mr_out_path}/invivo_npv_as_bf_img_{img_num}.mhd')
    # io.SaveITKFile(mr_day0_t2_slice, f'{day0_mr_out_path}/day0_t2_as_bf_img_{img_num}.mhd')
    # io.SaveITKFile(mr_day0_ctd_slice, f'{day0_mr_out_path}/day0_ctd_as_bf_img_{img_num}.mhd')
    # io.SaveITKFile(mr_day0_t1_slice, f'{day0_mr_out_path}/day0_ce_t1_as_bf_img_{img_num}.mhd')
    # io.SaveITKFile(mr_day0_npv_slice, f'{day0_mr_out_path}/day0_ce_t1_as_bf_img_{img_num}.mhd')

    del blockface, phi_inv, phi_inv_data
    torch.cuda.empty_cache()

    histology_seg = core.StructuredGrid.FromGrid(segs[0], channels=1)
    for seg in segs:
        histology_seg += seg

    ctd_slice = mr_log_ctd_slice.data[0].cpu()
    ctd_mask = np.logical_and(ctd_slice > 10.0, circle_mask == 1)
    masked_thermal = np.ma.masked_where(ctd_mask == 0.0, ctd_slice.numpy()).squeeze()
    plt.figure()
    # plt.imshow(t1_nc.data.cpu()[0, slice_n].squeeze(), aspect=1.0 / aspect, cmap='gray')
    plt.imshow(masked_thermal, cmap='jet', vmin=10, vmax=240)
    plt.gca().patch.set_facecolor([0, 0, 0, 0])
    plt.axis('off')

    zero_slice = np.zeros_like(ctd_slice)
    masked_slice = np.ma.masked_where(zero_slice == 0.0, zero_slice).squeeze()
    plt.figure()
    plt.imshow(masked_slice.squeeze(), cmap='gray')
    npv_contours = measure.find_contours(ctd_slice.data.squeeze().cpu().numpy() * circle_mask, 240)
    for contour in npv_contours:
        plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1.8)
    plt.axis('off')
    plt.gca().patch.set_facecolor([0, 0, 0, 0])
    plt.savefig(f'{invivo_mr_out_path}/day0_ctd_contour.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
    save = True

    # Generate the figures without masking
    generate_figures(
        mr_d0_t1_slice, segs, out_path=invivo_mr_out_path, base_name='day0_ce_t1_no_mask', save=save
    )

    generate_figures(
        mr_invivo_t1_slice, segs, out_path=invivo_mr_out_path, base_name='deformed_invivo_ce_t1_MR_no_mask', save=save
    )

    histology_image = histology_seg * deformed_histology
    mr_invivo_t1_slice = histology_seg * mr_invivo_t1_slice
    mr_d0_t1_slice = histology_seg * mr_d0_t1_slice
    # mr_invivo_t2_slice = histology_seg * mr_invivo_t2_slice
    # mr_day0_t2_slice = histology_seg * mr_day0_t2_slice
    # mr_day0_ctd_slice = histology_seg * mr_day0_ctd_slice
    # mr_day0_t1_slice = histology_seg * mr_day0_t1_slice
    # mr_day0_npv_slice = histology_seg * mr_day0_npv_slice

    # mr_day0_ctd_slice.data[mr_day0_ctd_slice.data < 0.5] = 0.0
    # mr_day0_ctd_slice.data[mr_day0_ctd_slice.data >= 0.5] = 1.0
    #
    # mr_day0_npv_slice.data[mr_day0_npv_slice.data < 0.5] = 0.0
    # mr_day0_npv_slice.data[mr_day0_npv_slice.data >= 0.5] = 1.0

    hist_map = core.StructuredGrid.FromGrid(segs[0])
    for i, seg in enumerate(segs, 1):
        hist_map = hist_map + (i * seg)

    if len(segs) == 3:
        color_map = ListedColormap(['k', 'crimson', 'lime', 'gold'])
        plt.figure()
        plt.imshow(hist_map.data.squeeze().cpu(), cmap=color_map)

    elif len(segs) == 2:
        color_map = ListedColormap(['k', 'crimson', 'lime'])
        plt.figure()
        plt.imshow(hist_map.data.squeeze().cpu(), cmap=color_map)

    else:
        color_map = ListedColormap(['k', 'crimson'])
        plt.figure()
        plt.imshow(hist_map.data.squeeze().cpu(), cmap=color_map)
    plt.axis('off')
    plt.show()
    plt.pause(1.0)


    if save:
        plt.savefig(f'{invivo_mr_out_path}/histology_segmentation_map.png', dpi=600, bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{day0_mr_out_path}/histology_segmentation_map.png', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.close('all')

    # Generate the figure for showing the NPV on the Day0 MRI CE T1
    # contours = measure.find_contours(mr_day0_npv_slice.data.squeeze().cpu().numpy(), 0.5)


    # Plot the image with contours
    # plt.figure()
    # plt.imshow(mr_day0_t1_slice.data.permute(1, 2, 0).squeeze().cpu(), cmap='gray')
    # for contour in contours:
    #     plt.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=0.8)
    # plt.axis('off')
    # plt.show()
    # plt.pause(1.0)
    # if save:
    #     plt.savefig(f'{invivo_mr_out_path}/day0_npv_with_contours.png', dpi=600, bbox_inches='tight', pad_inches=0)
    #
    # plt.close('all')

    # generate_figures(
    #     mr_day0_npv_slice, segs, out_path=day0_mr_out_path, base_name='deformed_npv', save=save,
    #     extra_cont=contours
    # )

    generate_figures(
        histology_image, segs, out_path=invivo_mr_out_path, base_name='deformed_histology', save=save
    )

    generate_figures(
        mr_invivo_t1_slice, segs, out_path=invivo_mr_out_path, base_name='deformed_invivo_ce_t1_MR', save=save
    )
    generate_figures(
        mr_d0_t1_slice, segs, out_path=invivo_mr_out_path, base_name='day0_ce_t1', save=save
    )

    # generate_figures(
    #     mr_invivo_t2_slice, segs, out_path=invivo_mr_out_path, base_name='deformed_invivo_t2_MR', save=save
    # )

    # generate_figures(
    #     mr_invivo_adc_slice, segs, out_path=invivo_mr_out_path, base_name='deformed_invivo_adc_MR', save=save
    # )

    # generate_figures(
    #     mr_day0_t2_slice, segs, out_path=day0_mr_out_path, base_name='deformed_day0_t2_MR', save=save
    # )

    # generate_figures(
    #     mr_day0_ctd_slice, segs, out_path=day0_mr_out_path, base_name='deformed_day0_ctd_MR', save=save
    # )
    #
    # generate_figures(
    #     mr_day0_t1_slice, segs, out_path=day0_mr_out_path, base_name='deformed_day0_ce_t1_MR', save=True
    # )
    npv_contours = measure.find_contours(mr_invivo_npv_slice.data.squeeze().cpu().numpy(), 0.5)
    zero_im = np.zeros_like(mr_invivo_npv_slice.data.squeeze().cpu().numpy())
    fig = plt.figure()
    # plt.imshow(mr_invivo_t1_slice.data.cpu().squeeze())
    ax = fig.add_subplot()
    ax.set_xlim([0, zero_im.shape[1]])
    ax.set_ylim([0, zero_im.shape[0]])
    for contour in npv_contours:
        plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[2], linewidth=3.0)
    ax.invert_yaxis()
    ax.set_aspect(1)
    plt.axis('off')
    plt.savefig(f'{invivo_mr_out_path}/NPV_contours.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

    npv_contours = measure.find_contours(mr_d0_npv_slice.data.squeeze().cpu().numpy(), 0.5)
    zero_im = np.zeros_like(mr_d0_npv_slice.data.squeeze().cpu().numpy())
    fig = plt.figure()
    # plt.imshow(mr_invivo_t1_slice.data.cpu().squeeze())
    ax = fig.add_subplot()
    ax.set_xlim([0, zero_im.shape[1]])
    ax.set_ylim([0, zero_im.shape[0]])
    for contour in npv_contours:
        plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[0], linewidth=3.0)
    ax.invert_yaxis()
    ax.set_aspect(1)
    plt.axis('off')
    plt.savefig(f'{invivo_mr_out_path}/D0_NPV_contours.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

    if len(segs) == 3:
        hst_contours = measure.find_contours(segs[-1].data.squeeze().cpu().numpy(), 0.5)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlim([0, zero_im.shape[1]])
        ax.set_ylim([0, zero_im.shape[0]])
        for contour in hst_contours:
            plt.plot(contour[:, 1], contour[:, 0], color=matplotlib._cm._tab10_data[6], linewidth=3.0)
        ax.invert_yaxis()
        ax.set_aspect(1)
        plt.axis('off')

        plt.savefig(f'{invivo_mr_out_path}/HST_contours.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

    plt.close('all')
    print('Done')


if __name__ == '__main__':
    rabbit = '18_062'

    histology_dir = f'/hdscratch/ucair/{rabbit}/microscopic/'
    block_list = sorted(glob.glob(f'/hdscratch/ucair/{rabbit}/microscopic/block*'))
    for block_path in block_list[3:]:
        block = block_path.split("/")[-1]
        img_list = sorted(glob.glob(f'/hdscratch/ucair/{rabbit}/microscopic/{block}/raw/*image.jpg'))
        img_list = [x.split('/')[-1].split('_')[1] for x in img_list]
        for img in ['015']:
            blockface_slice = int(img)
            sample_on_histopathology(rabbit, block, img, blockface_slice)
