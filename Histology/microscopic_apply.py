import os
import sys
sys.path.append("/home/sci/blakez/code/")
import glob
import yaml
import copy
import h5py
import tools
import torch
import numpy as np
import subprocess as sp
import skimage.segmentation as seg

from collections import OrderedDict
import torch.optim as optim

import CAMP.camp.Core as core
import CAMP.camp.FileIO as io
import CAMP.camp.StructuredGridTools as st
import CAMP.camp.UnstructuredGridOperators as uo
import CAMP.camp.StructuredGridOperators as so

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def read_mhd_header(filename):

    with open(filename, 'r') as in_mhd:
        long_string = in_mhd.read()

    short_strings = long_string.split('\n')
    key_list = [x.split(' = ')[0] for x in short_strings]
    value_list = [x.split(' = ')[1] for x in short_strings]
    a = OrderedDict(zip(key_list, value_list))

    return a


def write_mhd_header(filename, dictionary):
    long_string = '\n'.join(['{0} = {1}'.format(k, v) for k, v in dictionary.items()])
    with open(filename, 'w+') as out:
        out.write(long_string)


def generate_label_volume(rabbit, block):
    data_dir = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'

    tif_files = sorted(glob.glob(f'{data_dir}volumes/raw/labels_tif/*'))

    hdr = read_mhd_header(f'{data_dir}volumes/raw/difference_volume.mhd')
    hdr['ElementDataFile'] = hdr['ElementDataFile'].replace('difference', 'segmentation')
    hdr['ElementNumberOfChannels'] = 1

    # Load each of the tif images and then dump it into the label dir
    for tif_file in tif_files:
        tif_name = tif_file.split('/')[-1].split('.')[0]

        if os.path.exists(f'{data_dir}volumes/raw/segmentation/{tif_name}.mhd'):
            continue

        tif = io.LoadITKFile(tif_file)

        io.SaveITKFile(tif, f'{data_dir}volumes/raw/segmentation/{tif_name}.mhd')

    write_mhd_header(f'{data_dir}volumes/raw/segmentation_volume.mhd', hdr)


def def_block(rabbit, block):
    data_dir = f'/hdscratch/ucair/microscopic/{rabbit}/registered/{block}/'
    bf_dir = f'/hdscratch/ucair/{rabbit}/blockface/{block}/'
    vol_ext = '/volumes/raw/'
    raw_ext = '/surfaces/raw/'
    def_ext = '/volumes/deformable/'

    # Load the affine
    aff = np.loadtxt(f'{bf_dir}{raw_ext}{block}_rigid_tform.txt')
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    # Load phi_inv
    phi_inv = io.LoadITKFile(f'{bf_dir}{vol_ext}{block}_phi_inv.mhd', device=device)
    phi_inv.set_size((60, 1024, 1024))
    #
    # # Phi_inv is in z,y,x and need x,y,z
    phi_inv.data = phi_inv.data.flip(0)

    # Load the volume to apply this field to
    mic = io.LoadITKFile(f'{data_dir}{block}_histopathology_affine_volume.mhd', device=device)
    # seg = io.LoadITKFile(f'{data_dir}{vol_ext}segmentation_volume.mhd', device=device)

    # aff_grid = core.StructuredGrid.FromGrid(phi_inv, channels=3)
    # aff_grid.set_to_identity_lut_()
    # aff_grid.data = aff_grid.data.flip(0)

    # Apply the inverse affine to the grid
    aff = aff.inverse()
    a = aff[0:3, 0:3].float()
    t = aff[-0:3, 3].float()

    # aff_grid.data = torch.matmul(a, aff_grid.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    # aff_grid.data = (aff_grid.data.squeeze() + t).permute([3] + list(range(0, 3)))
    #
    # aff_grid.data = aff_grid.data.flip(0)
    #
    # aff_bf = so.ApplyGrid.Create(aff_grid, device=aff_grid.device, dtype=aff_grid.dtype)(bf, out_grid=phi_inv)

    phi_inv.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                phi_inv.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    phi_inv.data = (phi_inv.data.squeeze() + t).permute([-1] + list(range(0, 3)))

    # Flip phi_inv back to the way it was
    phi_inv.data = phi_inv.data.flip(0)

    # try:
    #     stitch_files = sorted(glob.glob(f'{data_dir}{vol_ext}{block}_phi_inv_stitch_*.mhd'))
    #     stitch_list = []
    #     for stitch_file in stitch_files:
    #         phi_stitch = io.LoadITKFile(stitch_file, device=device)
    #         phi_stitch.set_size((60, 1024, 1024))
    #         stitch_list += [phi_stitch.clone()]
    #     composer = so.ComposeGrids.Create(padding_mode='border', device=device)
    #     final_phi = composer([phi_inv] + stitch_list)
    #     phi_inv = final_phi.clone()
    # except IOError:
    #     pass

    def_mic = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(mic, phi_inv)
    # def_seg = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(seg, phi_inv)

    # Save out the deformed volumes
    io.SaveITKFile(def_mic, f'{data_dir}{block}_histopathology_volume_deformable.mhd')
    # io.SaveITKFile(def_seg, f'{data_dir}{def_ext}segmentation_volume_deformable.mhd')


def compose_blocks(rabbit):
    data_dir = f'/hdscratch/ucair/microscopic/{rabbit}/registered/'
    bf_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    old_dir = f'/home/sci/blakez/ucair/{rabbit}/rawVolumes/ExVivo*/'
    def_ext = '/volumes/deformable/'
    block_list = [x.split('/')[-1] for x in sorted(glob.glob(f'{bf_dir}/block*'))]

    # Load the affine
    aff = np.loadtxt(glob.glob(f'{old_dir}blocks_to_exvivo_affine.txt')[0])
    aff = torch.tensor(aff, device=device, dtype=torch.float32)

    # Load the deformation
    phi_inv = io.LoadITKFile(glob.glob(f'{old_dir}block_phi_inv.mhd')[0], device=device)
    phi_inv.set_size((512, 512, 512))

    phi_inv.data = phi_inv.data.flip(0)

    # Apply the inverse affine to the grid
    aff = aff.inverse()
    a = aff[0:3, 0:3].float()
    t = aff[-0:3, 3].float()

    phi_inv.data = torch.matmul(a.unsqueeze(0).unsqueeze(0),
                                phi_inv.data.permute(list(range(1, 3 + 1)) + [0]).unsqueeze(-1))
    phi_inv.data = (phi_inv.data.squeeze() + t).permute([-1] + list(range(0, 3)))

    # Flip phi_inv back to the way it was
    phi_inv.data = phi_inv.data.flip(0)

    bf_list = []
    label_list = []

    for block in block_list:

        print(f'Deforming {block} ... ', end='')
        # if block == block_list[len(block_list) // 2]:
        #     bf = io.LoadITKFile(f'{data_dir}{block}/volumes/raw/difference_volume.mhd', device=device)
        #     # lb = io.LoadITKFile(f'{data_dir}{block}/volumes/raw/segmentation_volume.mhd', device=device)

        mic = io.LoadITKFile(f'{data_dir}{block}/{block}_histopathology_volume_deformable.mhd', device=device)
        # lb = io.LoadITKFile(f'{data_dir}{block}{def_ext}segmentation_volume_deformable.mhd', device=device)
        def_bf = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(mic, phi_inv)
        # def_lb = so.ApplyGrid.Create(phi_inv, pad_mode='zeros', device=device)(lb, phi_inv)
        def_bf.to_(device='cpu')
        # def_lb.to_(device='cpu')
        bf_list.append(def_bf.clone())
        # label_list.append(def_lb.clone())
        print('Done')

    lb_tensor = torch.stack([x.data for x in label_list], dim=0)
    lb_tensor = (lb_tensor > 0.45).float()

    bf_tensor = torch.stack([x.data for x in bf_list], dim=0)
    bf_sum = bf_tensor.sum(0)
    lb_sum = lb_tensor.sum(0)

    lb_non_zero = lb_sum.clone()
    lb_non_zero[lb_non_zero == 0.0] = 1.0

    bf_sum_scaled = bf_sum / lb_non_zero

    bf_to_exvivo = bf_list[0].clone()
    bf_to_exvivo.data = bf_sum_scaled.data.clone()

    io.SaveITKFile(bf_to_exvivo, '/home/sci/blakez/TEST_STACKED.mhd')

    print('Done')


def stack_hist(rabbit, block_list):
    data_dir = f'/hdscratch/ucair/microscopic/{rabbit}/registered/'
    bf_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    old_dir = f'/home/sci/blakez/ucair/{rabbit}/rawVolumes/ExVivo*/'

    image_count = 0

    # Load the very

    # Get a list of the files that are there
    for block in block_list:
        hdr_list = sorted(glob.glob(f'{data_dir}{block}/images/*.mhd'))
        raw_list = sorted(glob.glob(f'{data_dir}{block}/images/*.raw'))

        for hdr, raw in zip(hdr_list, raw_list):
            out_hdr = f'{data_dir}/recons/images/IMG_{image_count:03d}_histopathology_affine.mhd'



if __name__ == '__main__':
    rabbit = '18_047'
    data_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    block_list = [x.split('/')[-1] for x in sorted(glob.glob(f'{data_dir}/block*'))]

    # for block in block_list:
    #     print(f'Deforming {block} volumes ... ', end='')
    #     stack_hist(rabbit, block=block)
    #     print('Done')
    stack_hist(rabbit, block_list)
    compose_blocks(rabbit)
