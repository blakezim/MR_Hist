import os
import glob
import yaml
import tools

import CAMP.Core as core
import CAMP.FileIO as io

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def stitch_surfaces(rabbit):

    rabbit_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_ext = '/surfaces/raw/'
    vol_ext = '/volumes/raw/'

    # Get a list of the blocks
    block_list = sorted(glob.glob(f'{rabbit_dir}block*'))

    complete = ['block07', 'block09']

    for i, block_path in enumerate(block_list):

        block = block_path.split('/')[-1]
        target_surface_paths = sorted(glob.glob(f'{rabbit_dir}{block}{raw_ext}{block}_target_piece_surface*.obj'))
        source_surface_paths = sorted(glob.glob(f'{rabbit_dir}{block}{raw_ext}{block}_source_piece_surface*.obj'))

        target_surface_paths = [x for x in target_surface_paths if 'stitched' not in x]
        source_surface_paths = [x for x in source_surface_paths if 'stitched' not in x]

        if block in complete:
            print(f'{block} already registered ... ')
            continue

        if target_surface_paths == [] and source_surface_paths == []:
            print(f'No stitching surfaces for {block} ... ')
            continue

        source_surfaces = []
        target_surfaces = []

        for target_surface_path, source_surface_path in zip(target_surface_paths, source_surface_paths):

            try:
                verts, faces = io.ReadOBJ(target_surface_path)
                tar_surface = core.TriangleMesh(verts, faces)
                tar_surface.to_(device)
                target_surfaces.append(tar_surface.copy())
            except IOError:
                print(f'The target stitching surface for {block} was not found ... skipping')
                continue

            try:
                verts, faces = io.ReadOBJ(source_surface_path)
                src_surface = core.TriangleMesh(verts, faces)
                src_surface.to_(device)
                src_surface.flip_normals_()
                source_surfaces.append(src_surface.copy())
            except IOError:
                print(f'The source stitching surface for {block} was not found ... skipping')
                continue

        for ts, ss in zip(target_surfaces[1:], source_surfaces[1:]):
            source_surfaces[0].add_surface_(ss.vertices, ss.indices)
            target_surfaces[0].add_surface_(ts.vertices, ts.indices)

        src_surface = source_surfaces[0].copy()
        tar_surface = target_surfaces[0].copy()

        try:
            with open(f'{rabbit_dir}{block}{raw_ext}{block}_middle_stitch_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [5.0],
                'propagation_sigma': None,
                'deformable_lr': [0.001],
                'converge': 0.05,
            }

        # Do the deformable registration
        def_src_surface, _ = tools.deformable_register(
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
            grid_device='cuda:0'
        )

        # Save the parameters for the deformation
        with open(f'{rabbit_dir}{block}{raw_ext}{block}_middle_stitch_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Create the middle surface by going half way to the registered surface
        new_verts = src_surface.vertices.clone() + ((def_src_surface.vertices - src_surface.vertices) * 0.5)
        mid_surface = src_surface.copy()
        mid_surface.vertices = new_verts.clone()
        mid_surface.calc_normals()
        mid_surface.calc_centers()

        # Load the extra surfaces that need to be deformed
        extras_paths = [
            f'{rabbit_dir}{block}{raw_ext}{block}_decimate.obj',
            f'{rabbit_dir}{block}{raw_ext}{block}_ext.obj',
        ]

        if i > 0:
            extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_foot.obj']
        if i < len(block_list):
            extras_paths += [f'{rabbit_dir}{block}{raw_ext}{block}_head.obj']

        # Check for support surfaces
        extras_paths += sorted(glob.glob(f'{rabbit_dir}{block}{raw_ext}{block}*support.obj'))

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

        try:
            with open(f'{rabbit_dir}{block}{raw_ext}{block}_stitch_config.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            params = {
                'currents_sigma': [6.0],
                'propagation_sigma': [1.9, 1.9, 0.5],
                'deformable_lr': [0.0006],
                'converge': 2.0,
                'grid_size': [25, 100, 100]
            }

        # Do the deformable registration
        def_src_surface, def_tar_surface, def_excess, phi, phi_inv = tools.stitch_surfaces(
            tar_surface.copy(),
            src_surface.copy(),
            mid_surface.copy(),
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
        )

        # Save the parameters for the deformation
        with open(f'{rabbit_dir}{block}{raw_ext}{block}_stitch_config.yaml', 'w') as f:
            yaml.dump(params, f)

        # Save the deformations
        io.SaveITKFile(phi_inv, f'{rabbit_dir}{block}{vol_ext}{block}_phi_inv_stitch.mhd')
        io.SaveITKFile(phi, f'{rabbit_dir}{block}{vol_ext}{block}_phi_stitch.mhd')

        # Save out the stitched surfaces
        out_path = f'{rabbit_dir}{block}{raw_ext}{block}'
        for extra_path, extra_surface in zip(extras_paths, def_excess):
            name = extra_path.split('/')[-1].split(f'{block}')[-1].replace('.', '_stitched.')
            io.WriteOBJ(extra_surface.vertices, extra_surface.indices, f'{out_path}{name}')

        print(f'Done stitching {block} ... ')


if __name__ == '__main__':
    rabbit = '18_047'
    stitch_surfaces(rabbit)
