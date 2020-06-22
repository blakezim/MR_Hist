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


def solve_affine(histology, blockface, img_num, out_dir, device='cpu'):

    try:
        opt_affine = np.loadtxt(f'{out_dir}/img_{img_num}_affine_to_blockface.txt')
        opt_affine = torch.tensor(opt_affine, device=device, dtype=torch.float32)

        optaff_filter = so.AffineTransform.Create(affine=opt_affine, device=device)
        aff_histopathology = optaff_filter(histology, blockface)
        return aff_histopathology, opt_affine

    except IOError:

        points = torch.tensor(
            tools.LandmarkPicker([histology[0].squeeze().cpu(), blockface[0].squeeze().cpu()]),
            dtype=torch.float32,
            device=device
        ).permute(1, 0, 2)

        # Change to real coordinates
        points *= torch.cat([histology.spacing[None, None, :], blockface.spacing[None, None, :]], 0)
        points += torch.cat([histology.origin[None, None, :], blockface.origin[None, None, :]], 0)

        aff_filter = so.AffineTransform.Create(points[1], points[0], device=device)

        affine = torch.eye(3, device=device, dtype=torch.float32)
        affine[0:2, 0:2] = aff_filter.affine
        affine[0:2, 2] = aff_filter.translation

    aff_mic_seg = aff_filter(histology, blockface)

    # Do some additional registration just to make sure it is in the right spot
    similarity = so.L2Similarity.Create(device=device)
    model = so.AffineIntensity.Create(similarity, device=device)

    # Create the optimizer
    optimizer = optim.SGD([
        {'params': model.affine, 'lr': 1.0e-11},
        {'params': model.translation, 'lr': 1.0e-12}], momentum=0.9, nesterov=True
    )

    energy = []
    for epoch in range(0, 1000):
        optimizer.zero_grad()
        loss = model(
            blockface.data, aff_mic_seg.data
        )
        energy.append(loss.item())

        print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')

        loss.backward()  # Compute the gradients
        optimizer.step()  #

        # if epoch >= 2:
        if epoch > 10 and np.mean(energy[-10:]) - energy[-1] < 0.01:
            break

    itr_affine = torch.eye(3, device=device, dtype=torch.float32)
    itr_affine[0:2, 0:2] = model.affine
    itr_affine[0:2, 2] = model.translation

    opt_affine = torch.matmul(itr_affine.detach(), affine)

    # Create a new resample filter to make sure everything works
    optaff_filter = so.AffineTransform.Create(affine=opt_affine, device=device)

    aff_histopathology = optaff_filter(histology, blockface)

    return aff_histopathology, opt_affine


def deformable_histology_to_blockface(histology, blockface, scales, steps, gauss=True, mic=None):
    deformation = blockface.clone()
    deformation.set_to_identity_lut_()
    deformation_list = []
    orig_histology = histology.clone()

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
        histology = gauss(histology)

    # Steps
    for s in scales:

        temp_mic = histology.clone()
        temp_block = blockface.clone()

        scale_source = temp_mic.set_size(histology.size // s, inplace=False)
        scale_target = temp_block.set_size(blockface.size // s, inplace=False)
        deformation = deformation.set_size(blockface.size // s, inplace=False)

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

            # temp_def = match.get_field().clone()
            # temp_def.set_size(mic.size, inplace=True)
            # def_mic = so.ApplyGrid.Create(temp_def, device=device)(mic, temp_def)
            #
            # plt.figure()
            # plt.imshow(def_mic.data.permute(1, 2, 0).cpu())
            # plt.axis('off')
            # plt.gca().invert_yaxis()
            # plt.savefig(f'/home/sci/blakez/ucair/Animations/Scrolls/MicReg/Images/{i:03d}_image.png', dpi=500, bbox_inches='tight', pad_inches=0)
            # plt.close('all')
            # del temp_def, def_mic

            if i > 10 and np.mean(energy[-10:]) - energy[-1] < 0.001:
                break

        deformation = match.get_field()
        deformation_list.append(deformation.clone().set_size(histology.size, inplace=False))
        deformation = composer(deformation_list[::-1])

    # Compose the deformation fields
    source_def = so.ApplyGrid(deformation, device=device)(orig_histology, deformation)

    return source_def, deformation


def register_histopathology_to_blockface(rabbit, block, img_num, bf_slice):

    blockface_dir = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'
    histology_dir = f'/hdscratch/ucair/{rabbit}/microscopic/{block}/'
    out_dir = f'{histology_dir}deformations/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.exists(f'{out_dir}/img_{img_num}_deformation_to_blockface.mhd'):
        return

    # Load and make the histopathology segmentation
    segs = []
    segs += [io.LoadITKFile(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_healthy_tissue.nrrd',
                                device=device)]
    if os.path.exists(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_ablated_region.nrrd'):
        segs += [io.LoadITKFile(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_ablated_region.nrrd',
                                device=device)]
    if os.path.exists(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_uncertain_region.nrrd'):
        segs += [io.LoadITKFile(f'{histology_dir}segmentations/IMG_{img_num}/img_{img_num}_uncertain_region.nrrd',
                                device=device)]

    histology_seg = core.StructuredGrid.FromGrid(segs[0], channels=1)
    for seg in segs:
        histology_seg += seg

    try:
        blockface_seg = io.LoadITKFile(f'{blockface_dir}volumes/raw/hd_labels/{block}_hdlabel_volume.nrrd',
                                       device=device)
    except:
        blockface_seg = io.LoadITKFile(f'{blockface_dir}volumes/raw/segmentation_volume.nrrd',
                                       device=device)
    # Extract the slice
    blockface_seg = blockface_seg.extract_slice(bf_slice - 1, dim=0)

    aff_seg, affine = solve_affine(histology_seg, blockface_seg, img_num, out_dir=out_dir, device=device)
    np.savetxt(f'{out_dir}/img_{img_num}_affine_to_blockface.txt', affine.cpu().numpy())

    #### Apply the affine to the image
    mic_file = f'{histology_dir}hdf5/{block}_img{img_num}_image.hdf5'

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
    aff_mic = so.AffineTransform.Create(affine=affine)(mic, blockface_seg)
    # plt.figure()
    # plt.imshow(aff_mic.data.permute(1,2,0).cpu())
    # plt.axis('off')
    # plt.gca().invert_yaxis()
    # plt.savefig(f'/home/sci/blakez/ucair/Animations/Scrolls/Mic/Images/{blockface_slice}_image.png', dpi=500, bbox_inches='tight', pad_inches=0)

    def_histology, deformation = deformable_histology_to_blockface(
        aff_seg,
        blockface_seg,
        steps=[0.005, 0.001],
        scales=[4, 1],
        gauss=True, 
        mic=aff_mic
    )

    # Save out the deformation
    io.SaveITKFile(deformation, f'{out_dir}/img_{img_num}_deformation_to_blockface.mhd')


if __name__ == '__main__':
    rabbit = '18_062'
    block = 'block05'
    raw_images = sorted(glob.glob(f'/hdscratch/ucair/{rabbit}/microscopic/{block}/raw/*_image.tif'))
    if raw_images == []:
        raw_images = mic_list = sorted(glob.glob(f'/hdscratch/ucair/{rabbit}/microscopic/{block}/raw/*_image.jpg'))
    cur_nums = [int(x.split('/')[-1].split('_')[1]) for x in raw_images]

    for im in cur_nums:
        img = f'{im:03d}'
        blockface_slice = im
        register_histopathology_to_blockface(rabbit, block, img, blockface_slice)
    # apply_deformation_to_histology(rabbit, block, img)
