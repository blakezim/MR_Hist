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

def diffusion(im, weight):

    grads = so.Gradient.Create(dim=2, device=device)(im)
    update = so.Divergence.Create(dim=2, device=device)(grads * weight)

    return update.data.squeeze()

def example():

    gamma = 0.2
    n_iters = 1000



    f = box_image(core.StructuredGrid((256, 256), device=device))
    d = box_image(core.StructuredGrid((256, 256), device=device), break_width=40, val=10.0)

    fluid_kernel = so.FluidKernel.Create(f, device=device)

    f = so.Gaussian.Create(channels=1, kernel_size=5, sigma=2, dim=2, device=device)(f)
    grads = so.Gradient.Create(dim=2, device=device)(f)

    smooth_grads = grads.clone()
    for itr in range(0, n_iters):

        x = core.StructuredGrid.FromGrid(f, tensor=smooth_grads[0].unsqueeze(0), channels=1)
        y = core.StructuredGrid.FromGrid(f, tensor=smooth_grads[1].unsqueeze(0), channels=1)
        x_update = gamma * diffusion(x, weight=d)
        y_update = gamma * diffusion(y, weight=d)

        smooth_grads += core.StructuredGrid.FromGrid(f, tensor=torch.stack((x_update, y_update), 0), channels=2)

    print('something')

    # x_loc, y_loc = torch.meshgrid(torch.arange(0, 256), torch.arange(0, 256))
    # plt.quiver(y_loc, x_loc, grads.data[1].cpu(), grads.data[0].cpu())
    # imgout = grads[1].squeeze().cpu().numpy().copy()

    # initialize some internal variables
    # deltaS = np.zeros_like(imgout)
    # deltaE = deltaS.copy()
    # NS = deltaS.copy()
    # EW = deltaS.copy()
    # gS = np.ones_like(imgout)
    # gE = gS.copy()

    # niter = 5000
    # kappa = 100
    # gamma = 0.2
    # step = (1., 1.)
    # option = 1
    #
    # for ii in np.arange(1, niter):
    #
    #     # calculate the diffs
    #     deltaS[:-1, :] = np.diff(imgout, axis=0)
    #     deltaE[:, :-1] = np.diff(imgout, axis=1)
    #
    #     # if 0 < sigma:
    #     #     deltaSf = flt.gaussian_filter(deltaS, sigma);
    #     #     deltaEf = flt.gaussian_filter(deltaE, sigma);
    #     # else:
    #     deltaSf = deltaS;
    #     deltaEf = deltaE;
    #
    #     # conduction gradients (only need to compute one per dim!)
    #     # if option == 1:
    #     #     gS = np.exp(-(deltaSf / kappa) ** 2.) / step[0]
    #     #     gE = np.exp(-(deltaEf / kappa) ** 2.) / step[1]
    #     # elif option == 2:
    #     #     gS = 1. / (1. + (deltaSf / kappa) ** 2.) / step[0]
    #     #     gE = 1. / (1. + (deltaEf / kappa) ** 2.) / step[1]
    #
    #     # update matrices
    #     E = d.data.squeeze().cpu().numpy() * deltaE
    #     S = d.data.squeeze().cpu().numpy() * deltaS
    #     # E = gE * deltaE
    #     # S = gS * deltaS
    #
    #     # subtract a copy that has been shifted 'North/West' by one
    #     # pixel. don't as questions. just do it. trust me.
    #     NS[:] = S
    #     EW[:] = E
    #
    #     is_nan = np.isnan(E)
    #     if is_nan.any():
    #         break
    #
    #     NS[1:, :] -= S[:-1, :]
    #     EW[:, 1:] -= E[:, :-1]
    #
    #     # update the image
    #     imgout += gamma * (NS + EW)
    #     # is_nan = np.isnan(NS)
    #     # if is_nan.any():
    #     #     break


    # print('something')

if __name__ == '__main__':
    example()
