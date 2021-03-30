import os
import sys
sys.path.append("/home/sci/blakez/code/")
sys.path.append("..")
import yaml
import glob
import tools
import torch
import numpy as np
import torch.optim as optim
import CAMP.camp.Core as core
import CAMP.camp.FileIO as io
import CAMP.camp.UnstructuredGridOperators as uo
import CAMP.camp.StructuredGridOperators as so

# from .. import tools

# device = 'cuda:3'
device = 'cuda:2'


def affine(tar_surface, src_surface, affine_lr=1.0e-07, translation_lr=1.0e-06, converge=1.0,
           spatial_sigma=[20.0], device='cpu'):

    init_translation = tar_surface.vertices.mean(dim=0) - src_surface.vertices.mean(dim=0)
    # no_trans = torch.zeros_like(init_translation)
    # init_affine = torch.tensor([
    #     [1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, -1]
    # ], device=device).float()

    # init_affine = torch.tensor([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ], device=device).float()

    # init_affine = torch.tensor([
    #     [0, -1, 0],
    #     [1, 0, 0],
    #     [0, 0, -1]
    # ], device=device).float()

    # affine = init_affine
    # # translation = no_trans
    # # translation = -torch.matmul(affine, src_surface.centers.mean(0)) + src_surface.centers.mean(0) + translation
    #
    # # Construct a single affine matrix
    # full_init_aff = torch.eye(len(affine) + 1)
    # full_init_aff[0:len(affine), 0:len(affine)] = affine.clone()
    # # full_init_aff[0:len(affine), len(affine)] = translation.clone().t()
    #
    # print(full_init_aff)
    #
    # aff_tfrom = uo.AffineTransformSurface.Create(full_init_aff, device=device)
    # src_surface = aff_tfrom(src_surface)
    # # #
    # # save_dir = f'/home/sci/blakez/ucair/18_060_Exvivo/'
    # # io.WriteOBJ(aff_source.vertices, aff_source.indices, f'{save_dir}/init_affine_blocks.obj')
    # init_angle = torch.tensor(0.0, dtype=torch.float32, device=device)

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
        # init_angle = model.angle.detach().clone()
        init_translation = model.translation.detach().clone()
        # print(init_affine)
        # print(init_translation)

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
    base_dir = f'/home/sci/blakez/ucair/{rabbit}_Exvivo/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    ff = False

    target_file = f'{base_dir}exvivo_tissue_decimate.obj'
    if ff:
        source_file = f'{base_dir}exterior_surfaces_ff_decimate.obj'
    else:
        source_file = f'{base_dir}exterior_surfaces_decimate.obj'
    # source_file = f'{base_dir}ExVivo_2018-07-26/registration/Front_Face_Recon.obj'

    save_dir = f'{base_dir}'

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    verts, faces = io.ReadOBJ(source_file)
    src_surface = core.TriangleMesh(verts, faces)
    src_surface.to_(device=device)

    verts, faces = io.ReadOBJ(target_file)
    tar_surface = core.TriangleMesh(verts, faces)
    tar_surface.to_(device=device)
    # tar_surface.flip_normals_()

    print('Starting Affine ... ')
    # Load or create the dictionary for registration
    try:
        if ff:
            with open(f'{save_dir}/affine_front_face_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        else:
            with open(f'{save_dir}/affine_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'spatial_sigma': [15.0],
            'affine_lr': 1.0e-07,
            'translation_lr': 1.0e-06,
            'converge': 1.0
        }

    try:
        if ff:
            aff = np.loadtxt(f'{save_dir}/blocks_to_exvivo_affine_front_face.txt')
            aff = torch.tensor(aff, device=device)
        else:
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
        if ff:
            # Save out the parameters
            with open(f'{save_dir}/affine_front_face_config.yaml', 'w') as f:
                yaml.dump(params, f)
            np.savetxt(f'{save_dir}/blocks_to_exvivo_affine_front_face.txt', aff.cpu().numpy())
        else:
            # Save out the parameters:
            with open(f'{save_dir}/affine_config.yaml', 'w') as f:
                yaml.dump(params, f)
            np.savetxt(f'{save_dir}/blocks_to_exvivo_affine.txt', aff.cpu().numpy())

    aff_tfrom = uo.AffineTransformSurface.Create(aff, device=device)
    aff_source = aff_tfrom(src_surface)

    io.WriteOBJ(aff_source.vertices, aff_source.indices, f'{save_dir}/affine_blocks_{rabbit}.obj')

    if ff:
        return

    # block_paths = sorted(glob.glob('/home/sci/blakez/ucair/deformable/*'))
    # extras_paths = [f'{path}/{path.split("/")[-1]}_deformable_to_exvivo.obj' for path in block_paths]
    extra_surfaces = []
    # for path in extras_paths:
    #     try:
    #         verts, faces = io.ReadOBJ(path)
    #     except IOError:
    #         extra_name = path.split('/')[-1]
    #         print(f'{extra_name} not found as an extra ... removing from list')
    #         _ = extras_paths.pop(extras_paths.index(path))
    #         continue
    #
    #     extra_surfaces += [core.TriangleMesh(verts, faces)]
    #     extra_surfaces[-1].to_(device)

    print('Starting Deformable ... ')
    try:
        with open(f'{save_dir}/deformable_config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        params = {
            'currents_sigma': [5.0, 0.5],
            'propagation_sigma': [20.0, 20.0, 20.0],
            'deformable_lr': [1.0e-05, 1.0e-05],
            'converge': 8.0,
            'grid_size': [38, 38, 38],
            'n_iters': 200
        }

    if type(params['deformable_lr']) is not list:
        params['deformable_lr'] = [params['deformable_lr']] * len(params['currents_sigma'])

    def_surface, def_excess, phi, phi_inv = tools.deformable_register(
        tar_surface.copy(),
        aff_source.copy(),
        src_excess=extra_surfaces,
        deformable_lr=params['deformable_lr'],
        currents_sigma=params['currents_sigma'],
        prop_sigma=params['propagation_sigma'],
        converge=params['converge'],
        grid_size=params['grid_size'],
        accu_forward=True,
        accu_inverse=True,
        device=device,
        grid_device='cuda:2',
        iters=params['n_iters']
    )

    # # Save out the parameters:
    with open(f'{save_dir}/deformable_config.yaml', 'w') as f:
        yaml.dump(params, f)

    io.SaveITKFile(phi_inv, f'{base_dir}blocks_phi_inv_to_exvivo.mhd')
    io.SaveITKFile(phi, f'{base_dir}blocks_phi_to_exvivo.mhd')
    io.WriteOBJ(def_surface.vertices, def_surface.indices, f'{base_dir}deformable_blocks_{rabbit}.obj')


if __name__ == '__main__':
    rabbit = 'ExVivoTests'
    register(rabbit)
