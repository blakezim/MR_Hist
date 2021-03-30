import os
import glob
import shutil

import CAMP.camp.Core as core
import CAMP.camp.FileIO as io
import CAMP.camp.Core.Display as dsp

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()

device = 'cuda:1'


def sort_and_label(in_dir, rabbit):

    out_base_dir = f'/hdscratch/ucair/{rabbit}/microscopic/'

    files = sorted(glob.glob(in_dir + '*'))
    label_files = [x for x in files if 'label' in x]
    image_files = [x for x in files if 'label' not in x]

    for im in range(0, len(image_files)):
        label = io.LoadITKFile(label_files[im])
        plt.figure()
        plt.imshow(label.data.permute(1, 2, 0) / 255.0)
        plt.axis('off')
        plt.pause(1)

        block_num = input('Please input the block number [xx]: ')
        if block_num == 'skip':
            continue
        else:
            block_num = int(block_num)
        depth = int(input('Please input the depth of the section [xxx]: '))

        image_num = int(depth / 50.0) + 1

        out_image_dir = f'{out_base_dir}block{block_num:02d}/raw/'
        out_label_dir = f'{out_base_dir}block{block_num:02d}/raw_labels/'
        if not os.path.exists(out_image_dir):
            os.makedirs(out_image_dir)
            os.makedirs(out_label_dir)

        out_label = f'IMG_{image_num:03d}_histopathology_label.tif'
        out_image = f'IMG_{image_num:03d}_histopathology_image.tif'

        shutil.copy(label_files[im], f'{out_label_dir}{out_label}')

        if os.path.exists(f'{out_image_dir}{out_image}'):
            pass
        else:
            shutil.copy(image_files[im], f'{out_image_dir}{out_image}')
        plt.close('all')


if __name__ == '__main__':
    rabbit = '18_060'
    base_dir = f'/hdscratch/ucair/microscopic/{rabbit}_Microscopic/tiff/'
    sort_and_label(base_dir, rabbit)
