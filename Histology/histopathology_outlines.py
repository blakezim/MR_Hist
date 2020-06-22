import os
import glob
import h5py
import torch
from skimage import measure
import numpy as np
import matplotlib.colors
import sklearn.mixture.gaussian_mixture as gmm

import CAMP.Core as core
import CAMP.FileIO as io

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cpu'


def draw_contours(rabbit):

    raw_mic_dir = f'/hdscratch/ucair/{rabbit}/microscopic/'
    block_list = sorted(glob.glob(f'{raw_mic_dir}/*'))

    for block_path in block_list[6:]:
        block = block_path.split('/')[-1]
        mic_list = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))
        img_nums = [x.split('/')[-1].split('_')[1] for x in mic_list]

        for img in img_nums[2:]:

            mic_file = f'{raw_mic_dir}{block}/segmentations/IMG_{img}/img_{img}_color.nii.gz'
            healthy_file = f'{raw_mic_dir}{block}/segmentations/IMG_{img}/img_{img}_healthy_tissue.nrrd'
            uncertain_file = f'{raw_mic_dir}{block}/segmentations/IMG_{img}/img_{img}_uncertain_region.nrrd'
            ablation_file = f'{raw_mic_dir}{block}/segmentations/IMG_{img}/img_{img}_ablated_region.nrrd'

            # meta_dict = {}
            # with h5py.File(mic_file, 'r') as f:
            #     mic = f['RawImage/ImageData'][:, ::10, ::10]
            #     for key in f['RawImage'].attrs:
            #         meta_dict[key] = f['RawImage'].attrs[key]

            # mic = core.StructuredGrid(
            #     mic.shape[1:],
            #     tensor=torch.tensor(mic, dtype=torch.float32, device=device),
            #     spacing=torch.tensor([10.0, 10.0], dtype=torch.float32, device=device),
            #     device=device,
            #     dtype=torch.float32,
            #     channels=3
            # )
            save = False
            contour_width = 1.0
            mic = io.LoadITKFile(mic_file, device=device)
            parts = [io.LoadITKFile(healthy_file, device=device)]
            if os.path.exists(uncertain_file):
                parts += [io.LoadITKFile(uncertain_file, device=device)]
            if os.path.exists(ablation_file):
                parts += [io.LoadITKFile(ablation_file, device=device)]

            # mic = (mic - mic.min()) / (mic.max() - mic.min())
            # labels = seg.slic(mic.data.cpu().permute(1, 2, 0).double().numpy(), 4, multichannel=True)

            part_contours = []
            for p in parts:
                part_contours += [measure.find_contours(p.data.squeeze().cpu().numpy(), 0.5)]

            plt.figure()
            plt.imshow(mic.data.permute(1, 2, 0).cpu())
            for contour in part_contours[0]:
                plt.plot(contour[:, 1], contour[:, 0], color='orange', linestyle='dashed', linewidth=contour_width)
            try:
                for contour in part_contours[1]:
                    plt.plot(contour[:, 1], contour[:, 0], 'b-.', linewidth=contour_width)
            except IndexError:
                pass

            try:
                for contour in part_contours[2]:
                    plt.plot(contour[:, 1], contour[:, 0], 'r:', linewidth=1.5)
            except IndexError:
                pass

            plt.axis('off')
            plt.show()
            plt.pause(1.0)

            out_path = f'{raw_mic_dir}{block}/segmentations/IMG_{img}/'
            if save:
                plt.savefig(f'{out_path}img_{img}_region_outlines.png', dpi=600, bbox_inches='tight', pad_inches=0)

            plt.figure()
            plt.imshow(mic.data.permute(1, 2, 0).cpu())
            plt.axis('off')

            if save:
                plt.savefig(f'{out_path}img_{img}_original.png', dpi=600, bbox_inches='tight', pad_inches=0)

            plt.close('all')
            print(f'Done with {block}: {img}')


if __name__ == '__main__':
    rabbit = '18_047'
    draw_contours(rabbit)
