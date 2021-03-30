import h5py
import math
import torch
import numpy as np
import torch.optim as optim
from collections import OrderedDict

import CAMP.camp.Core as core
import CAMP.camp.FileIO as io
import skimage.segmentation as seg
import CAMP.camp.StructuredGridOperators as so

from CAMP.camp.Core import *
from CAMP.camp.UnstructuredGridOperators import *
from CAMP.camp.StructuredGridOperators.UnaryOperators.ApplyGridFilter import ApplyGrid
from CAMP.camp.StructuredGridOperators.UnaryOperators.AffineTransformFilter import AffineTransform

# import matplotlib
#
# matplotlib.use('qt5agg')
# import matplotlib.pyplot as plt
#
# plt.ion()


def affine_register(tar_surface, src_surface, affine_lr=1.0e-06, translation_lr=1.0e-04, converge=0.01,
                    spatial_sigma=[0.5], rigid=True, device='cpu', plot=True):
    # Plot the surfaces
    if plot:
        [_, fig, ax] = PlotSurface(tar_surface.vertices, tar_surface.indices)
        [src_mesh, _, _] = PlotSurface(src_surface.vertices, src_surface.indices, fig=fig, ax=ax, color=[1, 0, 0])

    # Find the inital translation
    init_translation = (tar_surface.centers.mean(0) - src_surface.centers.mean(0)).clone()
    init_affine = torch.eye(3, device=device).float()

    for sigma in spatial_sigma:

        # Create some of the filters
        model = AffineCurrents.Create(
            tar_surface.normals,
            tar_surface.centers,
            sigma=sigma,
            init_affine=init_affine,
            init_translation=init_translation,
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

            if rigid:
                with torch.no_grad():
                    U, s, V = model.affine.clone().svd()
                    model.affine.data = torch.mm(U, V.transpose(1, 0))

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

    # Create affine applier filter and apply
    aff_tfrom = AffineTransformSurface.Create(full_aff, device=device)
    aff_source_head = aff_tfrom(src_surface)
    if plot:
        src_mesh.set_verts(aff_source_head.vertices[aff_source_head.indices].detach().cpu().numpy())

    return full_aff


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


def stitch_surfaces(tar_surface, src_surface, reference_surface, src_excess=None, deformable_lr=1.0e-04,
                    currents_sigma=None, prop_sigma=None, converge=0.3, grid_size=None,
                    accu_forward=False, accu_inverse=False, device='cpu', grid_device='cpu'):
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
        expansion = (vert_max - vert_min) * 0.1
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
        model = StitchingCurrents.Create(
            src_surface.copy(),
            tar_surface.copy(),
            reference_surface.copy(),
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
        for epoch in range(0, 200):
            optimizer.zero_grad()
            loss = model()

            print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f} ')
            energy.append(loss.item())

            loss.backward()  # Compute the gradients

            with torch.no_grad():

                # Create a single array of the gradients to be propagated
                concat_grad = torch.cat([model.src_vertices.grad, model.tar_vertices.grad])
                concat_vert = torch.cat([model.src_vertices, model.tar_vertices])

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
                    model.tar_vertices.grad = _prop_gradients(model.tar_vertices, concat_grad, concat_vert, prop_sigma)

                    optimizer.step()
                    if accu_forward:
                        phi = _update_phi(phi, optimizer.param_groups[2]['params'][0].clone())
                    if accu_inverse:
                        phi_inv = _update_phi_inv(phi_inv, identity, optimizer.param_groups[2]['params'][0].clone())

                    optimizer.param_groups[2]['params'][0].data = torch.zeros_like(identity.data).to(grid_device)

                else:

                    # Propagate the gradients to the register surfaces
                    model.src_vertices.grad = _prop_gradients(model.src_vertices, concat_grad, concat_vert, prop_sigma)
                    model.tar_vertices.grad = _prop_gradients(model.tar_vertices, concat_grad, concat_vert, prop_sigma)

                    optimizer.step()

            if epoch > 10 and np.mean(energy[-7:]) - energy[-1] < converge:
                break

        # Update the surfaces
        src_surface.vertices = model.src_vertices.detach().clone()
        tar_surface.vertices = model.tar_vertices.detach().clone()
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


def process_mic(micicroscopic, mic_seg_file, blockface, label, device='cpu'):
    meta_dict = {}

    try:
        with h5py.File(micicroscopic, 'r') as f:
            mic = f['ImageData'][1, ::10, ::10]
    except KeyError:
        with h5py.File(micicroscopic, 'r') as f:
            mic = f['RawImage/ImageData'][1, ::10, ::10]
            for key in f['RawImage'].attrs:
                meta_dict[key] = f['RawImage'].attrs[key]

    with h5py.File(mic_seg_file, 'r') as f:
        mic_seg = f['ImageData'][0, ::10, ::10]

    mic = core.StructuredGrid(
        mic.shape,
        tensor=torch.tensor(mic, dtype=torch.float32, device=device).unsqueeze(0),
        spacing=torch.tensor([10.0, 10.0], dtype=torch.float32, device=device),
        device=device,
        dtype=torch.float32,
        channels=1
    )

    mic_seg = core.StructuredGrid(
        mic_seg.shape,
        tensor=torch.tensor(mic_seg, dtype=torch.float32, device=device).unsqueeze(0),
        spacing=torch.tensor([10.0, 10.0], dtype=torch.float32, device=device),
        device=device,
        dtype=torch.float32,
        channels=1
    )
    # mic_seg.data = (mic_seg.data <= 0.5).float()

    if 'Affine' in meta_dict:
        opt_affine = torch.tensor(meta_dict['Affine'], dtype=torch.float32, device=device)
        optaff_filter = AffineTransform.Create(affine=opt_affine, device=device)
        aff_mic_image = optaff_filter(mic, blockface)
        aff_mic_label = optaff_filter(mic_seg, blockface)

        return aff_mic_image, aff_mic_label, opt_affine

    points = torch.tensor(
        LandmarkPicker([mic[0].cpu(), blockface[1].cpu()]),
        dtype=torch.float32,
        device=device
    ).permute(1, 0, 2)

    # Change to real coordinates
    points *= torch.cat([mic.spacing[None, None, :], blockface.spacing[None, None, :]], 0)
    points += torch.cat([mic.origin[None, None, :], blockface.origin[None, None, :]], 0)

    aff_filter = AffineTransform.Create(points[1], points[0], device=device)

    affine = torch.eye(3, device=device, dtype=torch.float32)
    affine[0:2, 0:2] = aff_filter.affine
    affine[0:2, 2] = aff_filter.translation

    # aff_mic_image = aff_filter(mic, blockface)
    aff_mic_seg = aff_filter(mic_seg, blockface)

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
            label.data, aff_mic_seg.data
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
    optaff_filter = AffineTransform.Create(affine=opt_affine, device=device)

    aff_mic_image = optaff_filter(mic, blockface)
    aff_mic_label = optaff_filter(mic_seg, blockface)

    return aff_mic_image, aff_mic_label, opt_affine


def LandmarkPicker(imList):

    '''Allows the user to select landmark correspondences between any number of images.
    The images must be in a list and must be formatted as numpy arrays. '''

    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()

    class PointPicker(object):
        '''Image class for picking landmarks'''

        def __init__(self, X):
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.im = self.ax.imshow(X, cmap='gray')
            self.shape = np.shape(X)
            self.cords = []
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        def onclick(self, event):
            if event.button == 3:
                xidx = int(round(event.xdata))
                yidx = int(round(event.ydata))
                self.cords.append([xidx, yidx])
                self.plot()

        def plot(self):
            self.ax.scatter(self.cords[-1][0], self.cords[-1][1])
            self.ax.set_xlim([0, self.shape[1]])
            self.ax.set_ylim([self.shape[0], 0])
            # plt.pause(0.001)

    plt.ion()
    pickerList = [PointPicker(im) for im in imList]
    plt.show()
    # while all([p.fig.get_visible() for p in pickerList]):
    Done = False
    while not Done:
        plt.pause(0.01)
        Done = plt.waitforbuttonpress()

    lmCoords = [p.cords for p in pickerList]
    lengths = [len(l) for l in lmCoords]
    if min(lengths) != max(lengths):
        raise Exception('Lists of landmarks were not consistent for each image, start over!')

    for p in pickerList:
        plt.close(p.fig)

    landmarks = np.array(lmCoords).swapaxes(0, 1).tolist()

    if len(pickerList) == 1:
        return landmarks

    for lm in landmarks:
        lm[0] = lm[0][::-1]
        lm[1] = lm[1][::-1]

    return landmarks


def read_mhd_header(filename):

    with open(filename, 'r') as in_mhd:
        long_string = in_mhd.read()

    short_strings = long_string.split('\n')
    short_strings = [x for x in short_strings if x != '']
    key_list = [x.split(' = ')[0] for x in short_strings]
    value_list = [x.split(' = ')[1] for x in short_strings]
    a = OrderedDict(zip(key_list, value_list))

    return a


def write_mhd_header(filename, dictionary):
    long_string = '\n'.join(['{0} = {1}'.format(k, v) for k, v in dictionary.items()])
    with open(filename, 'w+') as out:
        out.write(long_string)


if __name__ == '__main__':
    pass

    # Luminanace
    # lum = 0.2126 * mic.data[0] + 0.7152 * mic.data[1] + 0.0722 * mic.data[2]

