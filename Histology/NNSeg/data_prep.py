import os
import glob
import torch
import numpy as np
import SimpleITK as sitk


def preprocess_itk_image(image_path):
    itk_image = sitk.ReadImage(image_path)
    numpy_image = np.squeeze(sitk.GetArrayFromImage(itk_image))
    return torch.from_numpy(numpy_image)


def complile_training_data(rabbits):

    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()

    hd_root = '/hdscratch/ucair/'
    sd_root = '/usr/sci/scratch/blakez/microscopic_data/'

    if not os.path.exists(f'{sd_root}'):
        os.makedirs(f'{sd_root}')

    inputs = []
    labels = []

    for rabbit in rabbits:

        # Get the list of blocks that have segmentations
        rabbit_path = f'{hd_root}{rabbit}/'
        block_pwds = sorted(glob.glob(f'{rabbit_path}/microscopic/block*'))
        blocks = [x.split('/')[-1] for x in block_pwds]

        for block in blocks:
            print(f'===> Generating {rabbit}/{block} ... ', end='')

            # Get a list of the microscopic images
            section_list = sorted(glob.glob(f'{rabbit_path}/microscopic/{block}/segmentations/IMG_*'))

            for sec in section_list:
                img_num = sec.split('/')[-1].split('_')[-1]
                # If anything is going to exist, it will be the healthy tissue
                if not os.path.exists(f'{sec}/img_{img_num}_healthy_tissue.nrrd'):
                    continue

                segs = []
                segs += [preprocess_itk_image(f'{sec}/img_{img_num}_healthy_tissue.nrrd')]
                if os.path.exists(f'{sec}/img_{img_num}_uncertain_region.nrrd'):
                    segs += [preprocess_itk_image(f'{sec}/img_{img_num}_uncertain_region.nrrd')]
                if os.path.exists(f'{sec}/img_{img_num}_ablated_region.nrrd'):
                    segs += [preprocess_itk_image(f'{sec}/img_{img_num}_ablated_region.nrrd')]

                label = np.zeros(segs[0].shape)
                for i, seg in enumerate(segs, 1):
                    if i == 3:
                        label[seg.bool()] = 1
                    else:
                        label[seg.bool()] = i

                colors = []
                colors += [preprocess_itk_image(f'{sec}/img_{img_num}_red.nii.gz')]
                colors += [preprocess_itk_image(f'{sec}/img_{img_num}_green.nii.gz')]
                colors += [preprocess_itk_image(f'{sec}/img_{img_num}_blue.nii.gz')]

                hist_image = np.stack(colors, 0)

                inputs.append(torch.tensor(hist_image, dtype=torch.float32))
                labels.append(torch.tensor(label, dtype=torch.int8))
            print('done')

    torch.save(inputs, f'{sd_root}histology_inputs_two_label.pth')
    torch.save(labels, f'{sd_root}histology_labels_two_label.pth')

    print('Done')


if __name__ == '__main__':
    complile_training_data(['18_047', '18_062'])
