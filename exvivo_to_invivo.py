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

from collections import OrderedDict
import torch.optim as optim

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridTools as st
import CAMP.UnstructuredGridOperators as uo
import CAMP.StructuredGridOperators as so

# import matplotlib
# matplotlib.use('qt5agg')
# import matplotlib.pyplot as plt
# plt.ion()

device = 'cuda:1'


def affine(tar_surface, src_surface, affine_lr=1.0e-07, translation_lr=1.0e-06, converge=1.0,
           spatial_sigma=[20.0], device='cpu'):

    init_translation = tar_surface.vertices.mean(0) - src_surface.vertices.mean(0)
    init_affine = torch.eye(3, device=device).float()

    for sigma in spatial_sigma:

        # Create some of the filters
        model = uo.AffineCurrents.Create(
            tar_surface.normals,
            tar_surface.centers,
            init_affine=init_affine,
            init_translation=init_translation,
            sigma=sigma,
            kernel='cauchy',
            device=device
        )

        # Create the optimizer
        optimizer = optim.SGD([
            {'params': model.affine, 'lr': affine_lr},
            {'params': model.translation, 'lr': translation_lr}], momentum=0.9, nesterov=True
        )

        energy = [model.currents.e3.item()]
        for epoch in range(0, 200):
            optimizer.zero_grad()
            loss = model(
                src_surface.normals.clone(), src_surface.centers.clone()
            )

            print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')
            energy.append(loss.item())

            loss.backward()  # Compute the gradients
            optimizer.step()  # Apply the gradients

            if epoch > 10 and np.mean(energy[-10:]) - energy[-1] < converge:
                break

        # Update the affine and translation for the next sigma
        init_affine = model.affine.detach().clone()
        init_translation = model.translation.detach().clone()

    # # Need to update the translation to account for not rotation about the origin
    affine = model.affine.detach()
    translation = model.translation.detach()
    translation = -torch.matmul(affine, src_surface.centers.mean(0)) + src_surface.centers.mean(0) + translation

    # Construct a single affine matrix
    full_aff = torch.eye(len(affine) + 1)
    full_aff[0:len(affine), 0:len(affine)] = affine.clone()
    full_aff[0:len(affine), len(affine)] = translation.clone().t()

    return full_aff


def deformable_register(tar_surface, src_surface, spatial_sigma=[0.5], deformable_lr=1.0e-04,
                        smoothing_sigma=[20.0, 20.0, 20.0], converge=2.0, device='cpu',
                        phi_inv_size=[64, 64, 64], phi_device='cpu'):

    def _calc_vector_field(verts, grads, phi_inv, sigma):
        with torch.no_grad():
            # Flatten phi_inv
            flat_phi = phi_inv.data.clone().flatten(start_dim=1).permute(1, 0).flip(-1)
            flat_phi = ((flat_phi.unsqueeze(1) - verts.unsqueeze(0)) ** 2).sum(-1, keepdim=True)
            flat_phi = torch.exp(-flat_phi / (2 * sigma[None, None, :]))
            flat_phi = (grads[None, :, :].repeat(len(flat_phi), 1, 1) * flat_phi).sum(1)

            # Now change it back
            v_field = flat_phi.flip(-1).permute(1, 0).reshape(phi_inv.data.shape)
            v_field *= -1

            return v_field.clone().contiguous()

    def _update_phi_inv(phi_inv, update):
        with torch.no_grad():
            # Add the vfield to the identity
            phi_inv = so.ApplyGrid.Create(update, pad_mode='border',
                                          device=update.device, dtype=update.dtype)(phi_inv, update)

            return phi_inv

    # Define a grid size
    grid_size = torch.tensor(phi_inv_size, device=device, dtype=tar_surface.vertices.dtype)

    # Create a structured grid for PHI inverse - need to calculate the bounding box
    vert_min = src_surface.vertices.min(0).values
    vert_max = src_surface.vertices.max(0).values

    # Expand beyond the min so that we contain the entire surface - 10 % should be enough
    expansion = (vert_max - vert_min) * 0.5
    vert_min -= expansion
    vert_max += expansion

    # the verts are in (x,y,z) and we need (z,y,x) for volumes
    vert_min = vert_min.flip(0)
    vert_max = vert_max.flip(0)

    # Calculate the spacing
    spacing = (vert_max - vert_min) / grid_size

    phi_inv = core.StructuredGrid(
        grid_size, spacing=spacing, origin=vert_min, device=phi_device, dtype=torch.float32, requires_grad=False
    )
    phi = core.StructuredGrid(
        grid_size, spacing=spacing, origin=vert_min, device=phi_device, dtype=torch.float32, requires_grad=False
    )
    identity = core.StructuredGrid(
        grid_size, spacing=spacing, origin=vert_min, device=phi_device, dtype=torch.float32, requires_grad=False
    )
    phi_inv.set_to_identity_lut_()
    phi.set_to_identity_lut_()
    identity.set_to_identity_lut_()

    smoothing_sigma = torch.tensor(smoothing_sigma, device=device)

    for sigma in spatial_sigma:

        # Create the deformable model
        model = uo.DeformableCurrents.Create(
            src_surface.copy(),
            tar_surface,
            sigma=sigma,
            kernel='cauchy',
            device=device
        )

        # Create a smoothing filter
        gauss = uo.GaussianSmoothing(smoothing_sigma, dim=3, device=device)

        # Set up the optimizer
        optimizer = optim.SGD([
            {'params': [model.src_vertices, phi.data], 'lr': deformable_lr}], momentum=0.9, nesterov=True
        )

        # Now iterate
        energy = []
        for epoch in range(0, 1000):
            optimizer.zero_grad()
            loss = model()

            print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f} ')
            energy.append(loss.item())

            loss.backward()  # Compute the gradients

            with torch.no_grad():
                model_verts = model.src_vertices.clone().to(device=phi_device)
                model_grads = model.src_vertices.grad.clone().to(device=phi_device)

                # Calcuate the vector field for the grid and put into identity grad
                optimizer.param_groups[0]['params'][1].grad = _calc_vector_field(
                    model_verts, model_grads, phi_inv, smoothing_sigma.clone().to(device=phi_device)
                )
            # Now the gradients are stored in the parameters being optimized
            model.src_vertices.grad = gauss(model.src_vertices)
            optimizer.step()  #

            with torch.no_grad():
                phi.data = optimizer.param_groups[0]['params'][1].data

                # Now that the grads have been applied to the identity field, we can use it to sample phi_inv
                phi_inv = _update_phi_inv(phi_inv, phi)
                # print((optimizer.param_groups[0]['params'][2].data - identity.data).max())
                # Set the optimizer data back to identity
                optimizer.param_groups[0]['params'][1].data = identity.data.clone()

            if epoch > 10 and np.mean(energy[-7:]) - energy[-1] < converge:
                break

        # Update the surfaces
        src_surface.vertices = model.src_vertices.detach().clone()

    return src_surface, phi_inv


def register(rabbit):
    target_file = '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/tumor_and_ablation_t1_exvivo.obj'
    source_file = '/home/sci/blakez/ucair/18_047/rawVolumes/PostImaging_2018-07-02/tumor_and_ablation_t1.obj'
    save_dir = '/hdscratch/ucair/18_047/mri/invivo/'

    verts, faces = io.ReadOBJ(source_file)
    invivo_surface = core.TriangleMesh(verts, faces)
    invivo_surface.to_(device=device)

    verts, faces = io.ReadOBJ(target_file)
    exvivo_surface = core.TriangleMesh(verts, faces)
    exvivo_surface.to_(device=device)

    print('Starting Affine ... ')
    # Load or create the dictionary for registration
    try:
        with open(f'{save_dir}surfaces/raw/affine_config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'spatial_sigma': [20.0],
            'affine_lr': 1.0e-08,
            'translation_lr': 1.0e-05,
            'converge': 1.0
        }

    try:
        aff = np.loadtxt(f'{save_dir}surfaces/raw/exvivo_to_invivo_affine.txt')
        aff = torch.tensor(aff, device=device)
    except IOError:
        aff = affine(
            invivo_surface.copy(), #Target
            exvivo_surface.copy(),
            affine_lr=params['affine_lr'],
            translation_lr=params['translation_lr'],
            converge=params['converge'],
            spatial_sigma=params['spatial_sigma'],
            device=device
        )
        # Save out the parameters:
        with open(f'{save_dir}surfaces/raw/affine_config.yaml', 'w') as f:
            yaml.dump(params, f)
        np.savetxt(f'{save_dir}surfaces/raw/exvivo_to_invivo_affine.txt', aff.cpu().numpy())

    aff_tfrom = uo.AffineTransformSurface.Create(aff, device=device)
    aff_exvivo = aff_tfrom(exvivo_surface)

    io.WriteOBJ(aff_exvivo.vertices, aff_exvivo.indices,
                f'{save_dir}surfaces/affine/exvivo_to_invivo_{rabbit}_affine.obj')

    print('Starting Deformable ... ')
    try:
        with open(f'{save_dir}/deformable_config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'spatial_sigma': [5.0, 2.0],
            'smoothing_sigma': [10.0, 10.0, 10.0],
            'deformable_lr': [1.0e-04, 1.0e-04],
            'converge': 0.5,
            'phi_inv_size': [32, 32, 32]
        }

    def_surface, _, phi, phi_inv = tools.deformable_register(
        invivo_surface.copy(),
        aff_exvivo.copy(),
        src_excess=None,
        deformable_lr=params['deformable_lr'],
        currents_sigma=params['spatial_sigma'],
        prop_sigma=params['smoothing_sigma'],
        converge=params['converge'],
        grid_size=params['phi_inv_size'],
        accu_forward=True,
        accu_inverse=True,
        device=device,
        grid_device='cuda:0',
        expansion_factor=1.0
    )

    # Save out the parameters:
    with open(f'{save_dir}surfaces/raw/deformable_config.yaml', 'w') as f:
        yaml.dump(params, f)

    io.SaveITKFile(phi_inv, f'{save_dir}volumes/deformable/invivo_phi_inv.mhd')
    io.SaveITKFile(phi, f'{save_dir}volumes/deformable/invivo_phi.mhd')
    io.WriteOBJ(def_surface.vertices, def_surface.indices,
                f'{save_dir}surfaces/deformable/exvivo_to_invivo_deformable.obj')


def apply_to_images(rabbit):

    affine = False
    data_dir = '/hdscratch/ucair/18_047/mri/invivo/'

    # Load in the file to be deformed
    ce_t1 = io.LoadITKFile(
        '/home/sci/blakez/ucair/18_047/rawVolumes/PostImaging_2018-07-02/012_----_3D_VIBE_0.5x0.5x1_NoGrappa_3avg_fatsat_cor.nii.gz',
        device=device
    )
    t2_motion = io.LoadITKFile(
        '/home/sci/blakez/ucair/18_047/elastVolumes/day3_motion/008_m--e_t2_spc_1mm_iso_cor.nii.gz',
        device=device
    )

    label = io.LoadITKFile(
        '/home/sci/blakez/ucair/18_047/rawVolumes/PostImaging_2018-07-02/T2_ablation_segmentation.nrrd',
        device=device
    )

    # Load the deformation field
    phi_inv = io.LoadITKFile(
        f'{data_dir}volumes/deformable/invivo_phi_inv.mhd', device=device
    )
    phi_inv.set_size((256, 256, 256))
    if affine:
        phi_inv.set_to_identity_lut_()
    phi_inv.data = phi_inv.data.flip(0)

    # Load the affine
    aff = np.loadtxt(f'{data_dir}surfaces/raw/invivo_to_exvivo_affine.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    # Apply the inverse affine to the grid
    aff = aff.inverse()
    a = aff[0:3, 0:3].float()
    t = aff[-0:3, 3].float()

    phi_inv.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                phi_inv.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    phi_inv.data = (phi_inv.data.squeeze() + t).permute([-1] + list(range(0, 3)))

    # Flip phi_inv back to the way it was
    phi_inv.data = phi_inv.data.flip(0)

    if affine:
        # Just apply the affine to the images
        aff_ce_t1 = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(ce_t1, phi_inv)
        aff_t2_motion = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(t2_motion, phi_inv)
        aff_label = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(label, phi_inv)
        io.SaveITKFile(aff_ce_t1, f'{data_dir}volumes/affine/invivo_to_exvivo_ce_t1_affine.mhd')
        io.SaveITKFile(aff_t2_motion, f'{data_dir}volumes/affine/invivo_to_exvivo_t2_affine.mhd')
        io.SaveITKFile(aff_label, f'{data_dir}volumes/affine/invivo_to_exvivo_label_affine.mhd')
    else:
        def_ce_t1 = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(ce_t1, phi_inv)
        def_t2_motion = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(t2_motion, phi_inv)
        def_label = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(label, phi_inv)
        # Save out the deformed volumes
        io.SaveITKFile(def_ce_t1, f'{data_dir}volumes/deformable/invivo_to_exvivo_ce_t1_deformable.mhd')
        io.SaveITKFile(def_t2_motion, f'{data_dir}volumes/deformable/invivo_to_exvivo_t2_deformable.mhd')
        io.SaveITKFile(def_label, f'{data_dir}volumes/deformable/invivo_to_exvivo_label_deformable.mhd')


if __name__ == '__main__':
    rabbit = '18_047'
    register(rabbit)
    # apply_to_images(rabbit)
