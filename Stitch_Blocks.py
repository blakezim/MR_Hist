import os
import glob
import yaml
import tools

import CAMP.Core as core
import CAMP.FileIO as io
import CAMP.StructuredGridOperators as so

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'



def deformable_register(tar_surface, src_surface, src_excess=None, deformable_lr=1.0e-04,
                        currents_sigma=None, prop_sigma=None, converge=0.3, grid_size=None,
                        accu_forward=False, accu_inverse=False, device='cpu', grid_device='cpu',
                        expansion_factor=0.1, iters=200):
    if currents_sigma is None:
        currents_sigma = [0.5]
    if prop_sigma is None:
        prop_sigma = [1.5, 1.5, 0.5]
    if grid_size is None:
        grid_size = [30, 100, 100]
    if src_excess is None:
        src_excess = []

    def _update_phi(phi, update_tensor):
        update = core.StructuredGrid.FromGrid(phi, tensor=update_tensor, channels=phi.channels)
        applier = ApplyGrid.Create(phi, pad_mode='border', device=update.device, dtype=update.dtype)
        return phi - applier(update)

    def _update_phi_inv(phi_inv, identity, update_tensor):
        update = core.StructuredGrid.FromGrid(phi_inv, tensor=update_tensor, channels=phi_inv.channels)
        smaple = identity.clone() + update

        return ApplyGrid.Create(smaple, pad_mode='border', device=update.device, dtype=update.dtype)(phi_inv)

    def _prop_gradients(prop_locations, grads, verts, prop_sigma):
        d = ((prop_locations.unsqueeze(1) - verts.unsqueeze(0)) ** 2).sum(-1, keepdim=True)
        d = torch.exp(-d / (2 * prop_sigma[None, None, :] ** 3))
        return (grads[None, :, :].repeat(len(d), 1, 1) * d).sum(1)

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

        return StructuredGrid(
            grid_size, spacing=spacing, origin=vert_min, device=grid_device, dtype=torch.float32, requires_grad=False
        )

    deformation = []

    if accu_forward or accu_inverse:
        identity = _create_grid(src_surface, src_excess, grid_size, grid_device)
        identity.set_to_identity_lut_()
        deformation.append(torch.zeros_like(identity.data).to(grid_device))
        if accu_forward:
            phi = StructuredGrid.FromGrid(identity)
            phi.set_to_identity_lut_()

        if accu_inverse:
            phi_inv = StructuredGrid.FromGrid(identity)
            phi_inv.set_to_identity_lut_()

    # Create the list of variable that need to be optimized
    extra_params = []
    for surface in src_excess:
        extra_params += [surface.vertices]

    prop_sigma = torch.tensor(prop_sigma, device=device)

    for i, sigma in enumerate(currents_sigma):

        # Create the deformable model
        model = DeformableCurrents.Create(
            src_surface.copy(),
            tar_surface.copy(),
            sigma=sigma,
            kernel='cauchy',
            device=device
        )

        # Set up the optimizer
        optimizer = optim.SGD([
            {'params': [model.src_vertices], 'lr': deformable_lr[i]},
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

                # Create a single array of the gradients to be propagated
                concat_grad = model.src_vertices.grad.clone()
                concat_vert = model.src_vertices.clone()

                # Need to propegate the gradients to the other vertices
                for surf in optimizer.param_groups[1]['params']:
                    # Propagate the updates from the vertices
                    surf.grad = _prop_gradients(surf, concat_grad, concat_vert, prop_sigma)

                if accu_forward or accu_inverse:

                    grid_grads = _prop_gradients(
                        identity.data.clone().flatten(start_dim=1).permute(1, 0).flip(-1).to(grid_device),
                        concat_grad.clone().to(grid_device),
                        concat_vert.clone().to(grid_device),
                        prop_sigma.clone().to(grid_device)
                    )

                    grid_grads = -1 * grid_grads.flip(-1).permute(1, 0).reshape(identity.shape()).contiguous()
                    optimizer.param_groups[2]['params'][0].grad = grid_grads.clone()

                    # Propagate the gradients to the register surfaces
                    model.src_vertices.grad = _prop_gradients(model.src_vertices, concat_grad, concat_vert, prop_sigma)

                    optimizer.step()
                    if accu_forward:
                        phi = _update_phi(phi, optimizer.param_groups[2]['params'][0].clone())
                    if accu_inverse:
                        phi_inv = _update_phi_inv(phi_inv, identity, optimizer.param_groups[2]['params'][0].clone())

                    optimizer.param_groups[2]['params'][0].data = torch.zeros_like(identity.data).to(grid_device)

                else:

                    # Propagate the gradients to the register surfaces
                    model.src_vertices.grad = _prop_gradients(model.src_vertices, concat_grad, concat_vert, prop_sigma)

                    optimizer.step()

            if epoch > 10 and np.mean(energy[-7:]) - energy[-1] < converge:
                break

        # Update the surfaces
        src_surface.vertices = model.src_vertices.detach().clone()
        for surface, def_verts in zip(src_excess, optimizer.param_groups[1]['params']):
            surface.vertices = def_verts.detach().clone()

    if accu_forward and accu_inverse:
        return src_surface, src_excess, phi, phi_inv
    elif accu_forward:
        return src_surface, src_excess, phi
    elif accu_inverse:
        return src_surface, src_excess, phi_inv
    else:
        return src_surface, src_excess


def stitch_surfaces(rabbit):

    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_ext = '/surfaces/raw/'
    vol_ext = '/volumes/raw/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    # complete = ['block08']
    complete = []

    for i, block_path in enumerate(block_list):

        block = block_path.split('/')[-1]
        stitching_dir = f'{rabbit_dir}{block}{raw_ext}/stitching/'

        if block in complete:
            continue

        if not os.path.exists(stitching_dir):
            print(f'No stitching surfaces found for {block}.')
            continue

        target_surface_path = f'{stitching_dir}/raw/{block}_target_piece_surface.obj'
        source_surface_paths = sorted(glob.glob(f'{stitching_dir}/raw/{block}_source_piece_surface*.obj'))

        # Load the target surface
        try:
            verts, faces = io.ReadOBJ(target_surface_path)
            tar_surface = core.TriangleMesh(verts, faces)
            tar_surface.to_(device)
        except IOError:
            print(f'The target stitching surface for {block} was not found ... skipping')
            continue

        source_surfaces = []
        # Load the source surface
        for source_surface_path in source_surface_paths:
            try:
                verts, faces = io.ReadOBJ(source_surface_path)
                src_surface = core.TriangleMesh(verts, faces)
                src_surface.to_(device)
                src_surface.flip_normals_()
                source_surfaces.append(src_surface.copy())
            except IOError:
                print(f'The source stitching surface for {block} was not found ... skipping')
                continue

        deformed_source_exts = []
        deformed_target_exts = []
        # Now we loop over the source pieces
        for i, source_surface in enumerate(source_surfaces):

            # Determine the surface half way between the source and the target
            params = {
                'currents_sigma': [1.0, 0.5],
                'propagation_sigma': [1.0, 1.0, 1.0],
                'deformable_lr': [0.001, 0.002],
                'converge': 0.05,
            }

            # Do the deformable registration
            def_src_surface, _ = tools.deformable_register(
                tar_surface.copy(),
                source_surface.copy(),
                src_excess=None,
                deformable_lr=params['deformable_lr'],
                currents_sigma=params['currents_sigma'],
                prop_sigma=params['propagation_sigma'],
                grid_size=None,
                converge=params['converge'],
                accu_forward=False,
                accu_inverse=False,
                device=device
            )

            new_verts = src_surface.vertices.clone() + ((def_src_surface.vertices - src_surface.vertices) * 0.5)
            mid_surface = src_surface.copy()
            mid_surface.vertices = new_verts.clone()
            mid_surface.calc_normals()
            mid_surface.calc_centers()

            # Need to load the exterior to drag along
            try:
                verts, faces = io.ReadOBJ(f'{stitching_dir}/raw/{block}_source_piece_ext_{i:02d}_decimate.obj')
                src_surface_ext = core.TriangleMesh(verts, faces)
                src_surface_ext.to_(device)
            except IOError:
                print(f'The source stitching surface for {block} was not found ... skipping')
                continue

            try:
                with open(f'{stitching_dir}/raw/{block}_stitch_config_{i:02d}.yaml', 'r') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)
            except IOError:
                params = {
                    'currents_sigma': [1.0, 0.5],
                    'propagation_sigma': [15.0, 15.0, 15.0],
                    'deformable_lr': [0.001, 0.002],
                    'converge': 0.05,
                }

            # Do the deformable registration with the source to the mid
            def_src_surface, def_src_ext = tools.deformable_register(
                mid_surface.copy(),
                source_surface.copy(),
                src_excess=[src_surface_ext.copy()],
                deformable_lr=params['deformable_lr'],
                currents_sigma=params['currents_sigma'],
                prop_sigma=params['propagation_sigma'],
                grid_size=None,
                converge=params['converge'],
                accu_forward=False,
                accu_inverse=False,
                device=device
            )

            # Need to load the target exterior to drag along
            try:
                verts, faces = io.ReadOBJ(f'{stitching_dir}/raw/{block}_target_piece_ext_decimate.obj')
                tar_surface_ext = core.TriangleMesh(verts, faces)
                tar_surface_ext.to_(device)
            except IOError:
                print(f'The source stitching surface for {block} was not found ... skipping')
                continue

            # Do the deformable registration with the source to the mid
            def_tar_surface, def_tar_ext = tools.deformable_register(
                mid_surface.copy(),
                tar_surface.copy(),
                src_excess=[tar_surface_ext.copy()],
                deformable_lr=params['deformable_lr'],
                currents_sigma=params['currents_sigma'],
                prop_sigma=params['propagation_sigma'],
                grid_size=None,
                converge=params['converge'],
                accu_forward=False,
                accu_inverse=False,
                device=device
            )

            # Save the parameters for the deformation
            with open(f'{stitching_dir}/raw/{block}_stitch_config_{i:02d}.yaml', 'w') as f:
                yaml.dump(params, f)

            # Keep track of the deformed exteriors
            deformed_source_exts += def_src_ext
            deformed_target_exts += def_tar_ext

            # Save out the deformed surfaces
            io.WriteOBJ(def_src_surface.vertices, def_src_surface.indices,
                        f'{stitching_dir}/deformable_pieces/{block}_source_piece_surface_{i:02d}.obj')
            io.WriteOBJ(def_src_ext[0].vertices, def_src_ext[0].indices,
                        f'{stitching_dir}/deformable_pieces/{block}_source_piece_ext_{i:02d}_decimate.obj')

            # Save out the deformed surfaces
            io.WriteOBJ(def_tar_surface.vertices, def_tar_surface.indices,
                        f'{stitching_dir}/deformable_pieces/{block}_target_piece_surface.obj')
            io.WriteOBJ(def_tar_ext[0].vertices, def_tar_ext[0].indices,
                        f'{stitching_dir}/deformable_pieces/{block}_target_piece_ext_decimate.obj')

        # Load the target exterior
        try:
            verts, faces = io.ReadOBJ(f'{stitching_dir}/raw/{block}_target_piece_ext_decimate.obj')
            tar_ext = core.TriangleMesh(verts, faces)
            tar_ext.to_(device)
        except IOError:
            print(f'The target stitching surface for {block} was not found ... skipping')
            continue

        # Generate the source surface
        source_ext_paths = sorted(glob.glob(f'{stitching_dir}/raw/{block}_source_piece_ext*_decimate.obj'))
        source_exts = []
        # Load the source surface
        for source_ext_path in source_ext_paths:
            try:
                verts, faces = io.ReadOBJ(source_ext_path)
                src_ext = core.TriangleMesh(verts, faces)
                src_ext.to_(device)
                source_exts.append(src_ext.copy())
            except IOError:
                print(f'The source stitching surface for {block} was not found ... skipping')
                continue

        src_whole_surface = tar_ext.copy()
        for surf in source_exts:
            src_whole_surface.add_surface_(surf.vertices, surf.indices)
        src_whole_surface.add_surface_(src_surface.vertices, src_surface.indices)
        src_whole_surface.add_surface_(tar_surface.vertices, tar_surface.indices)

        # Now that the stitched pieces have been generated, we need to construct the target
        tar_whole_surface = def_tar_ext[0].copy()
        for surf in deformed_source_exts:
            tar_whole_surface.add_surface_(surf.vertices, surf.indices)
        tar_whole_surface.add_surface_(def_tar_surface.vertices, def_tar_surface.indices)
        tar_whole_surface.add_surface_(def_src_surface.vertices, def_src_surface.indices)

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

        # Now do registration with the whole surfaces
        try:
            with open(f'{stitching_dir}/raw/{block}_stitch_config_large.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [5.0, 5.0],
                'propagation_sigma': [2.0, 2.0, 2.0],
                'deformable_lr': [0.00005, 0.0001],
                'converge': 0.2,
                'grid_size': [20, 64, 64]
            }

        # Do the deformable registration
        def_src_surface, def_extra_surfaces, phi1, phi_inv1 = tools.deformable_register(
            tar_whole_surface.copy(),
            src_whole_surface.copy(),
            src_excess=extra_surfaces,
            deformable_lr=params['deformable_lr'],
            currents_sigma=params['currents_sigma'],
            prop_sigma=params['propagation_sigma'],
            grid_size=params['grid_size'],
            converge=params['converge'],
            accu_forward=True,
            accu_inverse=True,
            device=device,
            grid_device='cuda:0',
            iters=400
        )

        # # Save the parameters for the deformation
        with open(f'{stitching_dir}/raw/{block}_stitch_config_large.yaml', 'w') as f:
            yaml.dump(params, f)

        try:
            with open(f'{stitching_dir}/raw/{block}_stitch_config_small.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [10.0, 10.0],
                'propagation_sigma': [0.5, 0.5, 0.5],
                'deformable_lr': [0.0001, 0.0002],
                'converge': 0.2,
                'grid_size': [20, 64, 64]
            }

        # Do the deformable registration
        def_src_surface, def_extra_surfaces, phi2, phi_inv2 = tools.deformable_register(
            tar_whole_surface.copy(),
            def_src_surface.copy(),
            src_excess=def_extra_surfaces,
            deformable_lr=params['deformable_lr'],
            currents_sigma=params['currents_sigma'],
            prop_sigma=params['propagation_sigma'],
            grid_size=params['grid_size'],
            converge=params['converge'],
            accu_forward=True,
            accu_inverse=True,
            device=device,
            grid_device='cuda:0',
            iters=400
        )

        with open(f'{stitching_dir}/raw/{block}_stitch_config_small.yaml', 'w') as f:
            yaml.dump(params, f)

        io.WriteOBJ(def_src_surface.vertices, def_src_surface.indices,
                    f'{stitching_dir}/deformable/{block}_whole_stitched_decimate.obj')

        phi_inv = so.ComposeGrids.Create(device='cuda:0')([phi_inv2, phi_inv1])
        phi = so.ComposeGrids.Create(device='cuda:0')([phi1, phi2])

        io.SaveITKFile(phi, f'{stitching_dir}/deformable/{block}_stitch_phi.mhd')
        io.SaveITKFile(phi_inv, f'{stitching_dir}/deformable/{block}_stitch_phi_inv.mhd')

        out_path = f'{stitching_dir}/deformable/'
        for extra_path, extra_surface in zip(extras_paths, def_extra_surfaces):
            name = extra_path.split('/')[-1]
            io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

        vol = io.LoadITKFile('/hdscratch/ucair/18_047/blockface/block08/volumes/raw/difference_volume.mhd',
                             device='cuda:0')

        phi_inv.set_size((60, 1024, 1024))
        phi.set_size((60, 1024, 1024))
        resampled_stitched = so.ApplyGrid.Create(phi_inv, device='cuda:0')(vol, phi_inv)
        resampled_unstitched = so.ApplyGrid.Create(phi, device='cuda:0')(resampled_stitched, phi)
        io.SaveITKFile(resampled_stitched, '/home/sci/blakez/stitched_block08.mhd')
        io.SaveITKFile(resampled_unstitched, '/home/sci/blakez/unstitched_block08.mhd')
        # # Save out the deformations

        print(f'Done stitching {block} ... ')


if __name__ == '__main__':
    rabbit = '18_047'
    stitch_surfaces(rabbit)