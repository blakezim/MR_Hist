import os
import glob
import yaml
import tools
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridOperators as so
from CAMP.UnstructuredGridOperators import *

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def deformable_register(tar_surface, src_surface, mid_surface, src_excess=None, deformable_lr=1.0e-04,
                        currents_sigma=None, prop_sigma=None, converge=0.3, grid_size=None,
                        accu_forward=False, accu_inverse=False, device='cpu', grid_device='cpu',
                        expansion_factor=0.1, iters=200, diff_params=None, mask=None):
    if currents_sigma is None:
        currents_sigma = [0.5]
    if prop_sigma is None:
        prop_sigma = [1.5, 1.5, 0.5]
    if grid_size is None:
        grid_size = [30, 100, 100]
    if src_excess is None:
        src_excess = []
    if diff_params is None:
        print('Must provide diffusion parameters.')
        exit()

    def _update_phi(phi, update_tensor):
        update = core.StructuredGrid.FromGrid(phi, tensor=update_tensor, channels=phi.channels)
        applier = so.ApplyGrid.Create(phi, pad_mode='border', device=update.device, dtype=update.dtype)
        return phi - applier(update)

    def _update_mask(mask, phi_inv):

        return so.ApplyGrid.Create(
            phi_inv, interp_mode='nearest', pad_mode='border', device=phi_inv.device, dtype=phi_inv.dtype
        )(mask)

    def _update_phi_inv(phi_inv, identity, update_tensor):
        update = core.StructuredGrid.FromGrid(phi_inv, tensor=update_tensor, channels=phi_inv.channels)
        smaple = identity.clone() + update

        return so.ApplyGrid.Create(smaple, pad_mode='border', device=update.device, dtype=update.dtype)(phi_inv)

    def _prop_gradients(prop_locations, grads, verts, prop_sigma):
        d = ((prop_locations.unsqueeze(1) - verts.unsqueeze(0)) ** 2).sum(-1, keepdim=True)
        d = torch.exp(-d / (2 * prop_sigma[None, None, :] ** 3))
        return (grads[None, :, :].repeat(len(d), 1, 1) * d).sum(1)

    def _diffuse_gradients(grads, mask, diff_params):

        orig_device = grads.device

        # generate the diffusion coefficient masks
        mask = mask.data.squeeze().clone()

        grads = grads.to('cuda:0')
        mask = mask.to('cuda:0')

        grads = grads * mask * diff_params['grad_amp']

        z_diffusion = mask.clone() * diff_params['z_diff_c'] + (diff_params['background_c'] / 10)
        y_diffusion = mask.clone() * diff_params['y_diff_c'] + diff_params['background_c']
        x_diffusion = mask.clone() * diff_params['x_diff_c'] + diff_params['background_c']

        diffused_gradients = torch.zeros_like(grads)

        for dir in range(0, len(grads)):

            print(f'=> Diffusing direction {dir}/2 gradients...', end='')

            imgout = grads[dir].squeeze().clone()

            # initialize some internal variables
            deltaD = torch.zeros_like(imgout)
            deltaS = deltaD.clone()
            deltaE = deltaD.clone()
            UD = deltaD.clone()
            NS = deltaD.clone()
            EW = deltaD.clone()

            for _ in np.arange(1, diff_params['niter']):

                # Calculate the gradients
                deltaD[:-1, :, :] = imgout[1:, :, :] - imgout[:-1, :, :]
                deltaS[:, :-1, :] = imgout[:, 1:, :] - imgout[:, :-1, :]
                deltaE[:, :, :-1] = imgout[:, :, 1:] - imgout[:, :, :-1]

                # Apply the diffusion coefficient
                D = z_diffusion * deltaD
                E = y_diffusion * deltaE
                S = x_diffusion * deltaS

                # Calculate divergence
                UD[:] = D
                NS[:] = S
                EW[:] = E
                UD[1:, :, :] -= D[:-1, :, :]
                NS[:, 1:, :] -= S[:, :-1, :]
                EW[:, :, 1:] -= E[:, :, :-1]

                # Update the image
                imgout += diff_params['gamma']*(UD+NS+EW)

            diffused_gradients[dir] = imgout.clone()

            print('done')

        return diffused_gradients.to(orig_device)

    def _sample_diffused_grads(grads, query_locs, mask, identity):

        surf_grads = grads.clone()
        surf_mask = mask.data.clone()
        surf_identity = identity.data.clone()

        surf_grads = surf_grads.to(query_locs.device)
        surf_mask = surf_mask.to(query_locs.device)
        surf_identity = surf_identity.to(query_locs.device)

        locs = surf_identity.flatten(start_dim=1).permute(1, 0).flip(-1)
        mask_flat = surf_mask.flatten().bool()
        grads_flat = surf_grads.flatten(start_dim=1).permute(1, 0).flip(-1)

        valid_locs = locs[mask_flat]
        valid_grads = grads_flat[mask_flat]

        out_grads = []
        for vert in query_locs:
            d = ((valid_locs - vert.unsqueeze(0)) ** 2).sum(-1)
            _, ind = torch.min(d, dim=0)
            out_grads.append(valid_grads[ind])

        out_grads = torch.stack(out_grads, 0)

        return out_grads

    def _create_grid(src_surface, src_excess, grid_size, grid_device):
        grid_size = torch.tensor(grid_size, device=device, dtype=tar_surface.vertices.dtype)
        extent_verts = src_surface.vertices.clone()

        for surface in src_excess:
            extent_verts = torch.cat([extent_verts, surface.vertices], 0)

        vert_min = extent_verts.min(0).values
        vert_max = extent_verts.max(0).values

        # Expand beyond the min so that we contain the entire surface - 10 % should be enough
        expansion = (vert_max - vert_min) * expansion_factor
        vert_min -= expansion
        vert_max += expansion

        # the verts are in (x,y,z) and we need (z,y,x) for volumes
        vert_min = vert_min.flip(0)
        vert_max = vert_max.flip(0)

        # Calculate the spacing
        spacing = (vert_max - vert_min) / grid_size

        return core.StructuredGrid(
            grid_size, spacing=spacing, origin=vert_min, device=grid_device, dtype=torch.float32, requires_grad=False
        )

    deformation = []

    # Neeed this when we are doing stitching
    identity = _create_grid(src_surface, src_excess, grid_size, grid_device)
    identity.set_to_identity_lut_()
    identity.to_(grid_device)
    deformation.append(torch.zeros_like(identity.data).to(grid_device))
    if accu_forward:
        phi = core.StructuredGrid.FromGrid(identity)
        phi.set_to_identity_lut_()

    if accu_inverse:
        phi_inv = core.StructuredGrid.FromGrid(identity)
        phi_inv.set_to_identity_lut_()

    # Resample the mask onto the grid
    mask.to_(grid_device)
    mask = so.ResampleWorld.Create(identity, device=grid_device)(mask)
    mask.data[mask.data > 0.0] = 1.0

    orig_mask = mask.copy()

    # Create the list of variable that need to be optimized
    extra_params = []
    for surface in src_excess:
        extra_params += [surface.vertices]

    smoothing_sigma = torch.tensor(prop_sigma, device=device)
    local_prop_sigma = torch.tensor(diff_params['propegation_sigma'], device=grid_device, dtype=torch.float32)

    for i, sigma in enumerate(currents_sigma):

        model = StitchingCurrents.Create(
            src_surface.copy(),
            tar_surface.copy(),
            mid_surface.copy(),
            sigma=sigma,
            kernel='cauchy',
            device=device
        )

        # Set up the optimizer
        optimizer = optim.SGD([
            {'params': [model.src_vertices, model.tar_vertices], 'lr': deformable_lr[i]},
            {'params': extra_params, 'lr': deformable_lr[i]},
            {'params': deformation, 'lr': deformable_lr[i]}], momentum=0.9, nesterov=True
        )

        # Now iterate
        energy = []
        for epoch in range(0, iters):
            optimizer.zero_grad()
            loss = model()

            print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f} ')
            energy.append(loss.item())

            loss.backward()  # Compute the gradients

            with torch.no_grad():

                # Propagate the gradients to the register surfaces
                smooth_src_grads = _prop_gradients(
                    model.src_vertices, model.src_vertices.grad, model.src_vertices, smoothing_sigma
                )

                smooth_tar_grads = _prop_gradients(
                    model.tar_vertices, model.tar_vertices.grad, model.tar_vertices, smoothing_sigma
                )

                # Create a single array of the gradients to be propagated
                concat_grad = torch.cat([smooth_src_grads, smooth_tar_grads], dim=0)
                concat_vert = torch.cat([model.src_vertices, model.tar_vertices], dim=0)

                grid_grads = _prop_gradients(
                    identity.data.clone().flatten(start_dim=1).permute(1, 0).flip(-1).to(grid_device),
                    concat_grad.clone().to(grid_device),
                    concat_vert.clone().to(grid_device),
                    local_prop_sigma
                )

                grid_grads = grid_grads.flip(-1).permute(1, 0).reshape(identity.shape()).contiguous()
                diffused_grads = _diffuse_gradients(grid_grads, mask, diff_params)

                sampled_src_grads = _sample_diffused_grads(diffused_grads, model.src_vertices, mask, identity)
                sampled_tar_grads = _sample_diffused_grads(diffused_grads, model.tar_vertices, mask, identity)

                # Need to propegate the gradients to the other vertices
                for surf in optimizer.param_groups[1]['params']:
                    # Propagate the updates from the vertices
                    surf.grad = _sample_diffused_grads(diffused_grads, surf, mask, identity)

                optimizer.param_groups[2]['params'][0].grad = -1 * diffused_grads.contiguous().clone()
                model.src_vertices.grad = sampled_src_grads
                model.tar_vertices.grad = sampled_tar_grads

                optimizer.step()
                if accu_forward:
                    phi = _update_phi(phi, optimizer.param_groups[2]['params'][0].clone())

                identity.set_to_identity_lut_()
                phi_inv = _update_phi_inv(phi_inv, identity, optimizer.param_groups[2]['params'][0].clone())
                optimizer.param_groups[2]['params'][0].data = torch.zeros_like(identity.data).to(grid_device)

                mask = _update_mask(orig_mask.clone(), phi_inv)

            if epoch > 10 and np.mean(energy[-7:]) - energy[-1] < converge:
                break
        # Update the surfaces
        tar_surface.vertices = model.tar_vertices.detach().clone()
        src_surface.vertices = model.src_vertices.detach().clone()
        for surface, def_verts in zip(src_excess, optimizer.param_groups[1]['params']):
            surface.vertices = def_verts.detach().clone()

    if accu_forward and accu_inverse:
        return src_surface, tar_surface, src_excess, phi, phi_inv
    elif accu_forward:
        return src_surface, tar_surface, src_excess, phi
    elif accu_inverse:
        return src_surface, tar_surface, src_excess, phi_inv
    else:
        return src_surface, tar_surface, src_excess


def stitch_surfaces(rabbit):

    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_ext = '/surfaces/raw/'
    vol_ext = '/volumes/raw/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    # complete = ['block08']
    complete = ['block06', 'block09']

    for i, block_path in enumerate(block_list):

        block = block_path.split('/')[-1]
        stitching_dir = f'{rabbit_dir}{block}{raw_ext}/stitching/'

        if block in complete:
            continue

        if not os.path.exists(stitching_dir):
            print(f'No stitching surfaces found for {block}.')
            continue

        target_surface_path = f'{stitching_dir}/raw/{block}_target_faces.obj'
        source_surface_path = f'{stitching_dir}/raw/{block}_source_faces.obj'

        # Load the target surface
        try:
            verts, faces = io.ReadOBJ(target_surface_path)
            tar_surface = core.TriangleMesh(verts, faces)
            tar_surface.to_(device)
        except IOError:
            print(f'The target stitching surface for {block} was not found ... skipping')
            continue

        try:
            verts, faces = io.ReadOBJ(source_surface_path)
            src_surface = core.TriangleMesh(verts, faces)
            src_surface.to_(device)
            src_surface.flip_normals_()
        except IOError:
            print(f'The source stitching surface for {block} was not found ... skipping')
            continue

        # Need to load the exterior to drag along
        try:
            verts, faces = io.ReadOBJ(f'{rabbit_dir}/{block}/{raw_ext}/{block}_decimate.obj')
            surface_ext = core.TriangleMesh(verts, faces)
            surface_ext.to_(device)
        except IOError:
            print(f'The source stitching surface for {block} was not found ... skipping')
            continue

        # Determine the surface half way between the source and the target
        try:
            with open(f'{stitching_dir}/raw/{block}_middle_surface_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [5.0, 0.25, 0.05],
                'propagation_sigma': [1.0, 1.0, 1.0],
                'deformable_lr': [0.0008, 0.01, 0.01],
                'converge': 0.05,
                'mid_offset': 0.5
            }

        # Do the deformable registration
        def_src_surface, def_ext = tools.deformable_register(
            tar_surface.copy(),
            src_surface.copy(),
            src_excess=None,
            deformable_lr=params['deformable_lr'],
            currents_sigma=params['currents_sigma'],
            prop_sigma=params['propagation_sigma'],
            grid_size=None,
            converge=params['converge'],
            accu_forward=False,
            accu_inverse=False,
            device=device,
        )

        new_verts = src_surface.vertices.clone() + ((def_src_surface.vertices - src_surface.vertices) * params['mid_offset'])
        mid_surface = src_surface.copy()
        mid_surface.vertices = new_verts.clone()
        mid_surface.calc_normals()
        mid_surface.calc_centers()

        io.WriteOBJ(def_src_surface.vertices, def_src_surface.indices,
                    f'{stitching_dir}/deformable_pieces/{block}_source_faces.obj')
        io.WriteOBJ(mid_surface.vertices, mid_surface.indices,
                    f'{stitching_dir}/deformable_pieces/{block}_source_middle.obj')

        with open(f'{stitching_dir}/raw/{block}_middle_surface_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Load the binary volume for the block
        mask = io.LoadITKFile(f'{rabbit_dir}/{block}/{vol_ext}/segmentation_volume.mhd', device='cuda:0')

        # Load the other surfaces to drag along
        extras_paths = [
            f'{rabbit_dir}{block}{raw_ext}{block}_decimate.obj',
            f'{rabbit_dir}{block}{raw_ext}{block}_ext.obj'
        ]

        if os.path.exists(f'{rabbit_dir}{block}{raw_ext}{block}_head.obj'):
            extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_head.obj']

        if os.path.exists(f'{rabbit_dir}{block}{raw_ext}{block}_foot.obj'):
            extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_foot.obj']

        if os.path.exists(f'{rabbit_dir}{block}{raw_ext}{block}_head_support.obj'):
            extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_head_support.obj']

        if os.path.exists(f'{rabbit_dir}{block}{raw_ext}{block}_foot_support.obj'):
            extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_foot_support.obj']

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

        # Define the diffusion parameters for the registration
        try:
            with open(f'{stitching_dir}/raw/{block}_diffusion_config.yaml', 'r') as f:
                diff_params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            diff_params = {
                'z_diff_c': 5.0,
                'y_diff_c': 200.0,
                'x_diff_c': 30.0,
                'background_c': 0.5,
                'niter': 10000,
                'gamma': 0.0005,
                'propegation_sigma': [0.2, 0.2, 0.2],
                'grad_amp': 300.0
            }

        try:
            with open(f'{stitching_dir}/raw/{block}_stitch_surface_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [2.0, 1.0],
                'smoothing_sigma': [2.0, 2.0, 2.0],
                'deformable_lr': [0.0001, 0.001],
                'grid_size': [40, 256, 256],
                'converge': 0.05,
                'grid_device': 'cpu',
                'niters': 25
            }

        # Do the deformable registration with the source to the mid
        def_src_surface, def_tar_surface, def_extra_surfaces, phi, phi_inv = deformable_register(
            tar_surface.copy(),
            src_surface.copy(),
            mid_surface.copy(),
            src_excess=extra_surfaces,
            deformable_lr=params['deformable_lr'],
            currents_sigma=params['currents_sigma'],
            prop_sigma=params['smoothing_sigma'],
            grid_size=params['grid_size'],
            converge=params['converge'],
            accu_forward=True,
            accu_inverse=True,
            grid_device=params['grid_device'],
            device=device,
            mask=mask,
            diff_params=diff_params,
            iters=params['niters']
        )

        with open(f'{stitching_dir}/raw/{block}_stitch_surface_config.yaml', 'w') as f:
            yaml.dump(params, f)

        with open(f'{stitching_dir}/raw/{block}_diffusion_config.yaml', 'w') as f:
            yaml.dump(diff_params, f)

        io.WriteOBJ(def_src_surface.vertices, def_src_surface.indices,
                    f'{stitching_dir}/deformable/{block}_whole_stitched_decimate.obj')

        io.SaveITKFile(phi, f'{stitching_dir}/deformable/{block}_stitch_phi.mhd')
        io.SaveITKFile(phi_inv, f'{stitching_dir}/deformable/{block}_stitch_phi_inv.mhd')

        out_path = f'{stitching_dir}/deformable/'
        for extra_path, extra_surface in zip(extras_paths, def_extra_surfaces):
            name = extra_path.split('/')[-1]
            io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

        vol = io.LoadITKFile('/hdscratch/ucair/18_047/blockface/block08/volumes/raw/difference_volume.mhd',
                             device='cuda:0')

        # phi_inv.set_size((60, 1024, 1024))
        # phi.set_size((60, 1024, 1024))
        # resampled_stitched = so.ApplyGrid.Create(phi_inv, device='cuda:0')(vol, phi_inv)
        # resampled_unstitched = so.ApplyGrid.Create(phi, device='cuda:0')(resampled_stitched, phi)
        # io.SaveITKFile(resampled_stitched, '/home/sci/blakez/stitched_block08.mhd')
        # io.SaveITKFile(resampled_unstitched, '/home/sci/blakez/unstitched_block08.mhd')

        print(f'Done stitching {block} ... ')


if __name__ == '__main__':
    rabbit = '18_047'
    stitch_surfaces(rabbit)

    # def _prop_gradients_normals(prop_locs, grads, grad_locs, grad_norms, prop_sigma):
    #     # Calculate the distance between where the current gradients are (grad_locs) and where we want them
    #     (prop_locs)
    #     d = ((prop_locs.unsqueeze(1) - grad_locs.unsqueeze(0)) ** 2).sum(-1, keepdim=True)
    #
    #     relation_norms = d.clone()
    #     relation_norms[relation_norms == 0.0] = 1.0
    #     relation_norms = ((prop_locs.unsqueeze(1) - grad_locs.unsqueeze(0)) / torch.sqrt(relation_norms))
    #     dot_prod = (relation_norms * grad_norms.unsqueeze(0).repeat(len(relation_norms), 1, 1)).sum(-1)
    #     dot_prod[dot_prod < 0.001] = 0.0
    #     # Calculate the normals
    #
    #     d = torch.exp(-d / (2 * prop_sigma[None, None, :] ** 3))
    #     return (grads[None, :, :].repeat(len(d), 1, 1) * d * dot_prod.unsqueeze(-1)).sum(1)
