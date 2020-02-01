import sys
sys.path.append("/home/sci/blakez/code/")
import yaml
import glob
import tools
import torch
import numpy as np
import torch.optim as optim
import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.UnstructuredGridOperators as uo
import CAMP.StructuredGridOperators as so

device = 'cuda:3'


def affine(tar_surface, src_surface, affine_lr=1.0e-07, translation_lr=1.0e-06, converge=1.0,
           spatial_sigma=[20.0], device='cpu'):

    init_translation = torch.tensor([0.0, 0.0, 0.0], device=device)
    init_affine = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ], device=device).float()

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


def register(rabbit):
    target_file = '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/Exvivo_surface_decimate.obj'
    source_file = '/home/sci/blakez/ucair/18_047/rawVolumes/ExVivo_2018-07-26/stacked_ext_deformable_decimate.obj'

    save_dir = str.join('/', target_file.split('/')[:-1])

    verts, faces = io.ReadOBJ(source_file)
    src_surface = core.TriangleMesh(verts, faces)
    src_surface.to_(device=device)

    verts, faces = io.ReadOBJ(target_file)
    tar_surface = core.TriangleMesh(verts, faces)
    tar_surface.to_(device=device)
    tar_surface.flip_normals_()

    print('Starting Affine ... ')
    # Load or create the dictionary for registration
    try:
        with open(f'{save_dir}/affine_config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'spatial_sigma': [20.0],
            'affine_lr': 1.0e-07,
            'translation_lr': 1.0e-06,
            'converge': 1.0
        }

    try:
        aff = np.loadtxt(f'{save_dir}/blocks_to_exvivo_affine.txt')
        aff = torch.tensor(aff, device=device)
    except IOError:

        aff = affine(
            tar_surface.copy(),
            src_surface.copy(),
            affine_lr=params['affine_lr'],
            translation_lr=params['translation_lr'],
            converge=params['converge'],
            spatial_sigma=params['spatial_sigma'],
            device=device
        )
        # Save out the parameters:
        # with open(f'{save_dir}/affine_config.yaml', 'w') as f:
        #     yaml.dump(params, f)
        # np.savetxt(f'{save_dir}/blocks_to_exvivo_affine.txt', aff.cpu().numpy())

    aff_tfrom = uo.AffineTransformSurface.Create(aff, device=device)
    aff_source = aff_tfrom(src_surface)

    # io.WriteOBJ(aff_source.vertices, aff_source.indices, f'{save_dir}/affine_blocks_{rabbit}.obj')

    block_paths = sorted(glob.glob('/home/sci/blakez/ucair/deformable/*'))
    extras_paths = [f'{path}/{path.split("/")[-1]}_deformable_to_exvivo.obj' for path in block_paths]
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

    print('Starting Deformable ... ')
    try:
        with open(f'{save_dir}/deformable_config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'currents_sigma': [5.0, 0.5],
            'propagation_sigma': [20.0, 20.0, 20.0],
            'deformable_lr': [1.0e-04],
            'converge': 8.0,
            'grid_size': [38, 38, 38]
        }

    if 'spatial_sigma' in params.keys():
        params['currents_sigma'] = params['spatial_sigma']
        del params['spatial_sigma']
    if 'phi_inv_size' in params.keys():
        params['grid_size'] = params['phi_inv_size']
        del params['phi_inv_size']
    if 'rigid_transform' in params.keys():
        del params['rigid_transform']
    if 'smoothing_sigma' in params.keys():
        params['propagation_sigma'] = params['smoothing_sigma']
        del params['smoothing_sigma']

    if type(params['deformable_lr']) is not list:
        params['deformable_lr'] = [params['deformable_lr']] * len(params['currents_sigma'])

    # def_surface, def_excess, phi, phi_inv = tools.deformable_register(
    #     tar_surface.copy(),
    #     aff_source.copy(),
    #     src_excess=extra_surfaces,
    #     deformable_lr=params['deformable_lr'],
    #     currents_sigma=params['currents_sigma'],
    #     prop_sigma=params['propagation_sigma'],
    #     converge=params['converge'],
    #     grid_size=params['grid_size'],
    #     accu_forward=True,
    #     accu_inverse=True,
    #     device=device,
    #     grid_device='cuda:3'
    # )
    def_surface, def_excess = tools.deformable_register(
            tar_surface.copy(),
            aff_source.copy(),
            src_excess=extra_surfaces,
            deformable_lr=params['deformable_lr'],
            currents_sigma=params['currents_sigma'],
            prop_sigma=params['propagation_sigma'],
            converge=params['converge'],
            grid_size=params['grid_size'],
            accu_forward=False,
            accu_inverse=False,
            device=device,
            grid_device='cuda:2'
        )

    out_path = f'/home/sci/blakez/ucair/to_exvivo/'
    for extra_path, extra_surface, block_path in zip(extras_paths, def_excess, block_paths):
        block = block_path.split('/')[-1]
        io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}/{block}/{block}_as_exvivo.obj')

    # Save out the parameters:
    # with open(f'{save_dir}/deformable_config.yaml', 'w') as f:
    #     yaml.dump(params, f)
    #
    # io.SaveITKFile(phi_inv, f'{save_dir}/blocks_phi_inv_to_exvivo.mhd')
    # io.SaveITKFile(phi, f'{save_dir}/blocks_phi_to_exvivo.mhd')
    # io.WriteOBJ(def_surface.vertices, def_surface.indices, f'{save_dir}/deformable_blocks_{rabbit}.obj')


if __name__ == '__main__':
    rabbit = '18_047'
    register(rabbit)
