import os
import glob
import h5py
import torch
import numpy as np
import matplotlib.colors
import sklearn.mixture.gaussian_mixture as gmm

import CAMP.Core as core
import CAMP.FileIO as io

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def cluster_image(mic, n_comps):
    hsv = matplotlib.colors.rgb_to_hsv(mic.data.cpu().permute(1, 2, 0).numpy())
    rgb = mic.data.cpu().permute(1, 2, 0).numpy()
    comb_data = np.concatenate([rgb, hsv], -1)
    newdata = comb_data.reshape(mic.data.shape[1] * mic.data.shape[2], 6)
    model = gmm.GaussianMixture(n_components=n_comps, covariance_type="full")
    model = model.fit(newdata)

    cluster = model.predict(newdata)
    cluster = cluster.reshape(mic.data.shape[1], mic.data.shape[2])

    return cluster


def process_mic(rabbit):

    raw_mic_dir = f'/hdscratch/ucair/{rabbit}/microscopic/'
    bf_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_bf_dir = f'/hdscratch/ucair/blockface/{rabbit}/'

    block_list = sorted(glob.glob(f'{raw_mic_dir}/*'))

    for block_path in block_list:
        block = block_path.split('/')[-1]

        mic_list = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))

        img_nums = [x.split('/')[-1].split('_')[1] for x in mic_list]

        for img in img_nums:

            if os.path.exists(f'{raw_mic_dir}{block}/segmentations/IMG_{img}/img_{img}_gmm_segmentation.nii.gz'):
                continue

            mic_file = f'{raw_mic_dir}{block}/hdf5/{block}_img{img}_image.hdf5'

            meta_dict = {}
            with h5py.File(mic_file, 'r') as f:
                mic = f['RawImage/ImageData'][:, ::10, ::10]
                for key in f['RawImage'].attrs:
                    meta_dict[key] = f['RawImage'].attrs[key]

            mic = core.StructuredGrid(
                mic.shape[1:],
                tensor=torch.tensor(mic, dtype=torch.float32, device=device),
                spacing=torch.tensor([10.0, 10.0], dtype=torch.float32, device=device),
                device=device,
                dtype=torch.float32,
                channels=3
            )

            mic = (mic - mic.min()) / (mic.max() - mic.min())
            # labels = seg.slic(mic.data.cpu().permute(1, 2, 0).double().numpy(), 4, multichannel=True)

            plt.figure()
            plt.imshow(mic.data.permute(1, 2, 0).cpu())
            plt.title('Microscopic Image')
            plt.show()
            plt.pause(1.0)

            satisfied = False

            while not satisfied:
                n_comps = int(input("Enter the number of components for segmentation: "))
                cluster = cluster_image(mic, n_comps)

                plt.figure()
                plt.imshow(cluster)
                plt.title('GMM Cluster')
                plt.colorbar()
                plt.show()
                plt.pause(1.0)


                redo_cluster = input("Are you satisfied with the clustering? [y/n]: ")
                if redo_cluster == 'y':
                    satisfied = True

            # Save out the original image and the segmentation
            out_path = f'{raw_mic_dir}{block}/segmentations/IMG_{img}/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            io.SaveITKFile(mic, f'{out_path}/img_{img}_color.nii.gz')
            io.SaveITKFile(
                core.StructuredGrid.FromGrid(mic, mic.data[0].unsqueeze(0)), f'{out_path}/img_{img}_red.nii.gz'
            )
            io.SaveITKFile(
                core.StructuredGrid.FromGrid(mic, mic.data[1].unsqueeze(0)), f'{out_path}/img_{img}_green.nii.gz'
            )
            io.SaveITKFile(
                core.StructuredGrid.FromGrid(mic, mic.data[2].unsqueeze(0)), f'{out_path}/img_{img}_blue.nii.gz'
            )
            io.SaveITKFile(
                core.StructuredGrid.FromGrid(mic, torch.tensor(cluster).unsqueeze(0)),
                f'{out_path}/img_{img}_gmm_segmentation.nii.gz'
            )

            plt.close('all')

            print(f'Done with {img}')


if __name__ == '__main__':
    rabbit = '18_047'
    process_mic(rabbit)
