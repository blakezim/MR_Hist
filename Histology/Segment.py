import os
import glob
import h5py
import torch
import numpy as np
import matplotlib.colors
import sklearn.mixture.gaussian_mixture as gmm

import CAMP.camp.Core as core
import CAMP.camp.FileIO as io

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


def convert_hdf5(block_path, raw_mic_dir):

    block = block_path.split('/')[-1]

    mic_list = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))
    # if mic_list == []:
    mic_list += sorted(glob.glob(f'{block_path}/raw/*_image.jpg'))

    img_nums = [x.split('/')[-1].split('_')[1] for x in mic_list]

    # Load the image
    for img in img_nums:
        print(f'Processing {block}, {img} ... ', end='')
        mic_file = f'{raw_mic_dir}{block}/hdf5/{block}_img{img}_image.hdf5'

        try:
            mic = io.LoadITKFile(f'{raw_mic_dir}{block}/raw/IMG_{img}_histopathology_image.tif')
        except RuntimeError:
            mic = io.LoadITKFile(f'{raw_mic_dir}{block}/raw/IMG_{img}_histopathology_image.jpg')

        with h5py.File(mic_file, 'w') as f:
            g = f.create_group('RawImage')
            g.create_dataset('ImageData', data=mic.data.numpy())

        print('Done')


def process_mic(rabbit):

    raw_mic_dir = f'/hdscratch/ucair/{rabbit}/microscopic/'
    bf_dir = f'/hdscratch/ucair/{rabbit}/blockface/'
    raw_bf_dir = f'/hdscratch/ucair/blockface/{rabbit}/'

    from Histology.NNSeg.models import UNet
    # from types import SimpleNamespace
    import torch.nn as nn
    from skimage import color
    model = UNet.UNet(6, 3)
    # saved_dict = SimpleNamespace(**torch.load(f'./Histology/NNSeg/model_weights/epoch_00230_model.pth'))

    # if opt.cuda:
    device = torch.device('cuda')
    model = model.to(device=device)
    model = nn.DataParallel(model)
    params = torch.load('./NNSeg/model_weights/epoch_00230_model.pth')
    model.load_state_dict(params['state_dict'])
    # else:
    #     device = torch.device('cpu')
    #     params = torch.load(f'./Histology/NNSeg/model_weights/epoch_00230_model.pth', map_location='cpu')
    #     model.load_state_dict(params['state_dict'])

    block_list = sorted(glob.glob(f'{raw_mic_dir}/block*'))

    for block_path in block_list:
        block = block_path.split('/')[-1]

        # Make sure that the hdf5 files have been written
        if not os.path.exists(f'{raw_mic_dir}{block}/hdf5/'):
            os.makedirs(f'{raw_mic_dir}{block}/hdf5/')
            convert_hdf5(block_path, raw_mic_dir)

        mic_list = sorted(glob.glob(f'{block_path}/raw/*_image.tif'))
        # if mic_list == []:
        mic_list += sorted(glob.glob(f'{block_path}/raw/*_image.jpg'))

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
                device='cpu',
                dtype=torch.float32,
                channels=3
            )

            mic = (mic - mic.min()) / (mic.max() - mic.min())

            plt.figure()
            plt.imshow(mic.data.permute(1, 2, 0).cpu())
            plt.title('Microscopic Image')
            plt.show()
            plt.pause(1.0)

            satisfied = False

            while not satisfied:
                n_comps = int(input("Enter the number of components for segmentation: "))
                print('Clustering ... ', end='')
                cluster = cluster_image(mic, n_comps)
                print('done')

                plt.figure()
                plt.imshow(cluster)
                plt.title('GMM Cluster')
                plt.colorbar()
                plt.show()
                plt.pause(1.0)

                redo_cluster = input("Are you satisfied with the clustering? [y/n]: ")
                if redo_cluster == 'y':
                    satisfied = True

            input_hsv = torch.from_numpy(color.rgb2hsv(mic.data.squeeze().permute(1, 2, 0))).permute(2, 0, 1)
            inputs = torch.cat([mic.data.squeeze(), input_hsv], dim=0)
            pred = model(inputs.unsqueeze(0)).cpu().detach().squeeze()

            seg = torch.zeros((pred.shape[1], pred.shape[2]))
            for i in range(0, pred.shape[0]):
                label = pred[i]
                seg[label > 0] = i

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
            io.SaveITKFile(
                core.StructuredGrid.FromGrid(mic, seg.unsqueeze(0)), f'{out_path}/img_nn_seg.nii.gz'
            )

            plt.close('all')

            print(f'Done with {img}')


if __name__ == '__main__':
    rabbit = '18_060'
    process_mic(rabbit)
