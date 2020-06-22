import os
import sys
sys.path.append("..")
sys.path.append("/home/sci/blakez/code/")
import yaml
import tools
import torch
import numpy as np

import torch.optim as optim

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.UnstructuredGridOperators as uo

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
        for epoch in range(0, 1000):
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


def register(rabbit):
    source_path = f'/hdscratch/ucair/{rabbit}/mri/exvivo/surfaces/raw/'
    target_path = f'/hdscratch/ucair/{rabbit}/mri/invivo/surfaces/raw/'

    source_file = f'{source_path}exvivo_ablation_region_decimate.obj'
    target_file = f'{target_path}invivo_ablation_region_decimate.obj'

    verts, faces = io.ReadOBJ(target_file)
    invivo_surface = core.TriangleMesh(verts, faces)
    invivo_surface.to_(device=device)

    verts, faces = io.ReadOBJ(source_file)
    exvivo_surface = core.TriangleMesh(verts, faces)
    exvivo_surface.to_(device=device)

    print('Starting Affine ... ')
    # Load or create the dictionary for registration
    try:
        with open(f'{source_path}affine_config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'spatial_sigma': [20.0],
            'affine_lr': 1.0e-08,
            'translation_lr': 1.0e-05,
            'converge': 1.5
        }

    try:
        aff = np.loadtxt(f'{source_path}exvivo_to_invivo_affine.txt')
        aff = torch.tensor(aff, device=device)
    except IOError:
        aff = affine(
            invivo_surface.copy(),
            exvivo_surface.copy(),
            affine_lr=params['affine_lr'],
            translation_lr=params['translation_lr'],
            converge=params['converge'],
            spatial_sigma=params['spatial_sigma'],
            device=device
        )
        # Save out the parameters:
        with open(f'{source_path}affine_config.yaml', 'w') as f:
            yaml.dump(params, f)
        np.savetxt(f'{source_path}exvivo_to_invivo_affine.txt', aff.cpu().numpy())

    aff_tfrom = uo.AffineTransformSurface.Create(aff, device=device)
    aff_exvivo = aff_tfrom(exvivo_surface)

    if not os.path.exists(f'{source_path}../affine/'):
        os.makedirs(f'{source_path}../affine/')

    io.WriteOBJ(aff_exvivo.vertices, aff_exvivo.indices,
                f'{source_path}../affine/exvivo_to_invivo_{rabbit}_affine.obj')

    print('Starting Deformable ... ')
    try:
        with open(f'{source_path}/deformable_config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'currents_sigma': [5.0, 2.0],
            'propagation_sigma': [10.0, 10.0, 10.0],
            'deformable_lr': [1.0e-04, 1.0e-04],
            'converge': 1.0,
            'phi_inv_size': [32, 32, 32],
            'n_iters': 500,
        }

    def_surface, _, phi, phi_inv = tools.deformable_register(
        invivo_surface.copy(),
        aff_exvivo.copy(),
        src_excess=None,
        deformable_lr=params['deformable_lr'],
        currents_sigma=params['currents_sigma'],
        prop_sigma=params['propagation_sigma'],
        converge=params['converge'],
        grid_size=params['phi_inv_size'],
        accu_forward=True,
        accu_inverse=True,
        device=device,
        grid_device='cuda:0',
        expansion_factor=1.5,
        iters=params['n_iters']
    )

    # Save out the parameters:
    with open(f'{source_path}deformable_config.yaml', 'w') as f:
        yaml.dump(params, f)

    if not os.path.exists(f'{source_path}../../volumes/raw//'):
        os.makedirs(f'{source_path}../../volumes/raw//')
    if not os.path.exists(f'{source_path}../deformable/'):
        os.makedirs(f'{source_path}../deformable/')

    io.SaveITKFile(phi_inv, f'{source_path}../../volumes/raw/exvivo_to_invivo_phi_inv.mhd')
    io.SaveITKFile(phi, f'{source_path}../../volumes/raw/exvivo_to_invivo_phi.mhd')
    io.WriteOBJ(def_surface.vertices, def_surface.indices,
                f'{source_path}../deformable/exvivo_to_invivo_deformable.obj')


if __name__ == '__main__':
    rabbit = '18_062'
    register(rabbit)

