import torch
import numpy as np
import torch.optim as optim

from CAMP.Core import *
from CAMP.UnstructuredGridOperators import *


def affine_register(tar_surface, src_surface, sigma=0.5, rigid=True, device='cpu'):

    # Plot the surfaces
    [_, fig, ax] = PlotSurface(tar_surface.vertices, tar_surface.indices)
    [src_mesh, _, _] = PlotSurface(src_surface.vertices, src_surface.indices, fig=fig, ax=ax, color=[1, 0, 0])

    # Find the inital translation
    translation = (tar_surface.centers.mean(0) - src_surface.centers.mean(0)).clone()

    # Create some of the filters
    model = AffineCurrents.Create(
        tar_surface.normals,
        tar_surface.centers,
        sigma=sigma,
        init_translation=translation,
        kernel='cauchy',
        device=device
    )

    # Create the optimizer
    optimizer = optim.SGD([
        {'params': model.affine, 'lr': 1.0e-06},
        {'params': model.translation, 'lr': 1.0e-04}], momentum=0.9, nesterov=True
    )

    energy = [model.currents.e3.item()]
    for epoch in range(0, 300):
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

        if epoch > 10 and np.mean(energy[-10:]) - energy[-1] < 0.01:
            break

    # # Need to update the translation to account for not rotation about the origin
    affine = model.affine.detach()
    translation = translation.detach()
    translation = -torch.matmul(affine, src_surface.centers.mean(0)) + src_surface.centers.mean(0) + translation

    # Construct a single affine matrix
    full_aff = torch.eye(len(affine) + 1)
    full_aff[0:len(affine), 0:len(affine)] = affine.clone()
    full_aff[0:len(affine), len(affine)] = translation.clone().t()

    # Create affine applier filter and apply
    aff_tfrom = AffineTransformSurface.Create(full_aff, device=device)
    aff_source_head = aff_tfrom(src_surface)
    src_mesh.set_verts(aff_source_head.vertices[aff_source_head.indices].detach().cpu().numpy())

    return full_aff


def deformable_register(tar_surface, src_surface, sigma, src_excess=None, device='cpu'):
    # Create the deformable model
    model = DeformableCurrents.Create(
        src_surface.copy(), tar_surface, sigma=sigma, kernel='cauchy', device=device
    )

    # Create a smoothing filter
    sigma = torch.tensor([0.5, 0.5, 5.0], device=device)
    gauss = GaussianSmoothing(sigma, dim=3, device=device)

    # Set up the optimizer
    optimizer = optim.SGD([
        {'params': [model.src_vertices, src_excess.vertices], 'lr': 5.0e-05}], momentum=0.9, nesterov=True
    )

    # Define a grid size
    grid_size = [128, 128, 30]

    # Create a structured grid for PHI inverse
    phi_inv = StructuredGrid(grid_size, device=device, dtype=torch.float32, requires_grad=False)

    # Now iterate
    for epoch in range(0, 1500):
        optimizer.zero_grad()
        loss = model()

        print(f'===> Iteration {epoch:3} Energy: {loss.item():.3f}')

        loss.backward()  # Compute the gradients

        # Need to propegate the gradients to the other vertices
        with torch.no_grad():
            d = ((src_excess.vertices.unsqueeze(1) - model.src_vertices.unsqueeze(0)) ** 2).sum(-1, keepdim=True)
            gauss_d = torch.exp(-d / (2 * sigma[None, None, :]))
            src_excess.vertices.grad = (model.src_vertices.grad[None, :, :].repeat(len(d), 1, 1) * gauss_d).sum(1)

        # Now the gradients are stored in the parameters being optimized
        model.src_vertices.grad = gauss(model.src_vertices)
        optimizer.step()  #

    return model.src_vertices, src_excess, phi_inv


def register_surfaces(tar_element, src_element, sigma, src_excess=None, device='cpu'):
    # Do the rigid registration of the
    affine_tform = affine_register(
        tar_element.copy(), src_element.copy(), rigid=True, device=device
    )

    # Do the deformable registration
    def_surface, def_excess, phi_inv = deformable_register(
        tar_element.copy(), src_element.copy(), sigma, src_excess=src_excess, device=device
    )


if __name__ == '__main__':
    pass
