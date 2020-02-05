import os
import sys
import glob
import yaml
import copy
import h5py
import tools
import torch
import shutil
import subprocess as sp
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

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def box_image(Im, break_width=50, center=None, val=1.0):

    if center == None:
        center = int(Im.size[0] // 2)

    Im.data[:, :, 0:(center - break_width // 2)] = val
    Im.data[:, :, (center + break_width // 2):] = val

    return Im


def diffuse(im, weight, iter=50000, gamma=0.0005):

    imgout = im.clone()

    # initialize some internal variables
    deltaS = torch.zeros_like(imgout)
    deltaE = deltaS.clone()
    NS = deltaS.clone()
    EW = deltaS.clone()
    # gS = torch.ones_like(imgout)

    for _ in np.arange(1, iter):
        deltaS[:-1, :] = imgout[1:, :] - imgout[:-1, :]
        deltaE[:, :-1] = imgout[:, 1:] - imgout[:, :-1]

        # update matrices
        E = weight * deltaE
        S = weight * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout


def example():

    f = box_image(core.StructuredGrid((256, 256), spacing=[1.0, 1.0], device=device))
    d = box_image(core.StructuredGrid((256, 256), spacing=[1.0, 1.0], device=device), break_width=50, val=500.0) + 0.1
    m = box_image(core.StructuredGrid((256, 256), spacing=[1.0, 1.0], device=device), break_width=50, val=1.0)

    f = so.Gaussian.Create(channels=1, kernel_size=5, sigma=2, dim=2, device=device)(f)
    grads = so.Gradient.Create(dim=2, device=device)(f) * m
    orig_grads = grads.copy()

    phi_inv = core.StructuredGrid((256, 256), spacing=[1.0, 1.0], device=device)
    phi_inv.set_to_identity_lut_()

    id = phi_inv.copy()

    for i in range(0, 6):

        diffuse_x_grad = grads[0].data
        diffuse_y_grad = diffuse(grads[1].data, d.data.squeeze())

        # Scale back up the gradients
        y_scale = (grads[1].max() - grads[1].min()) / (diffuse_y_grad.max() - diffuse_y_grad.min())

        update = core.StructuredGrid.FromGrid(
            f, tensor=torch.stack((diffuse_x_grad, y_scale * diffuse_y_grad), 0), channels=2
        )

        sample = id.clone() + 20.0 * update
        phi_inv = so.ApplyGrid.Create(sample, pad_mode='border', device=update.device, dtype=update.dtype)(phi_inv)

        # update the gradients
        f = so.ApplyGrid.Create(phi_inv, device=device)(f)
        d = so.ApplyGrid.Create(phi_inv, device=device)(d)
        m = so.ApplyGrid.Create(phi_inv, device=device)(m)
        grads = so.Gradient.Create(dim=2, device=device)(f) * m

        print(f'Iter {i}/5 done...')

    test = so.ApplyGrid.Create(phi_inv, device=device)(f)

    plt.figure()
    plt.imshow(d.cpu())
    plt.colorbar()
    plt.title('Diffusion Coefficient')

    plt.figure()
    plt.imshow(imgout.cpu())
    plt.colorbar()
    plt.title('Diffused Gradients')
    #
    plt.figure()
    plt.imshow(grads[1].squeeze().cpu())
    plt.colorbar()
    plt.title('Starting Gradients')
    #
    # def_grid = id.copy()
    # def_grid.set_to_identity_lut_()
    # def_grid += smooth_grads
    #
    # phi_inv = so.ApplyGrid.Create(def_grid, device=device)(phi_inv)
    #
    # def_image = so.ApplyGrid.Create(phi_inv, device=device)(f)
    #
    # plt.figure()
    # plt.imshow(def_image.data.squeeze().cpu() - f.data.squeeze().cpu())
    # plt.colorbar()
    # plt.title('Difference Of Images')
    #
    # plt.figure()
    # plt.imshow(def_image.data.squeeze().cpu())
    # plt.title('Deformed Image')

    print('something')

def example_fluid():

    f = box_image(core.StructuredGrid((256, 256), spacing=[0.05, 0.05], device=device))
    d = box_image(core.StructuredGrid((256, 256), spacing=[0.05, 0.05], device=device), break_width=46, val=100.0)

    f = so.Gaussian.Create(channels=1, kernel_size=5, sigma=2, dim=2, device=device)(f)
    grads = so.Gradient.Create(dim=2, device=device)(f) * 100

    fluid_kernel = so.FluidKernel.Create(f, device=device)

    # Flow the gradients

    # Compute the gradients of the x
    x = core.StructuredGrid.FromGrid(f, tensor=grads[1].unsqueeze(0), channels=1)
    x_grads = so.Gradient.Create(dim=2, device=device)(grads[1])

if __name__ == '__main__':
    example()
