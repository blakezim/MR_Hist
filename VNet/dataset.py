import glob
import torch
import numpy as np
import numbers
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib

# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt


def is_image_file(filename):
    """Check that the give file is an image."""
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    """Load an image from a given file path."""
    return Image.open(filepath)


# This is only going to take 1 cube from each volume currently
class TrainDataset(data.Dataset):

    def __init__(self, source_volume_list, label_volume_list, cube_size, length):
        super(TrainDataset, self).__init__()

        self.source = source_volume_list
        self.target = label_volume_list
        self.cubeSize = cube_size
        self.length = length

    def __getitem__(self, item):

        # Modulate the item so we get multiple cubes from a volume

        index = item % len(self.source)

        source = self.source[index]
        target = self.target[index]
        shape = source.shape

        y = torch.randint(0, (shape[2] - self.cubeSize[1]), (1,)).item()
        z = torch.randint(0, (shape[3] - self.cubeSize[2]), (1,)).item()

        source_cube = source[:, :, y: y + self.cubeSize[1], z: z + self.cubeSize[2]]
        target_cube = target[:, y: y + self.cubeSize[1], z: z + self.cubeSize[2]]

        return source_cube.float(), target_cube

    def __len__(self):
        return self.length


class EvalDataset(data.Dataset):
    def __init__(self, tensor_list, cube_size, stride=None):
        super(EvalDataset, self).__init__()

        self.cube = cube_size
        self.stride = cube_size
        if stride:
            self.stride = (np.array(cube_size) // np.array(stride)).tolist()

        for itr, source in enumerate(tensor_list):
            # Need to pad dimensions
            # This z padding might be a bit extreme in some cases
            shape = source.shape
            z_pad = self.cube[0] - (shape[1] % self.cube[0])
            x_pad = self.cube[1] - (shape[2] % self.cube[1])
            y_pad = self.cube[2] - (shape[3] % self.cube[2])

            # replication pad is not yet released for n-dimensional

            pad = (y_pad // 2, y_pad - (y_pad // 2),
                   x_pad // 2, x_pad - (x_pad // 2))
            source = F.pad(source, pad, "replicate")

            # Now pad the front and back
            front = source[:, 0, :, :].unsqueeze(1).repeat(1, z_pad // 2, 1, 1)
            back = source[:, -1, :, :].unsqueeze(1).repeat(1, z_pad - (z_pad // 2), 1, 1)
            source = torch.cat((front, source, back), 1)

            pad_shape = source.shape[1:4]

            # x_blocks = int(source.shape[2] / self.cube[1])
            # y_blocks = int(source.shape[3] / self.cube[2])

            source = source.unfold(1, self.cube[0], self.stride[0]).unfold(2, self.cube[1], self.stride[1]).unfold(3, self.cube[2], self.stride[2])
            num_blocks = source.shape[1:4]
            source = source.permute(1, 2, 3, 0, 4, 5, 6)
            source = source.contiguous().view(-1, 3, self.cube[0], self.cube[1], self.cube[2])
            # else:
            #
            #     # Consider doing this with striding as well
            #     front = source[:, 0:self.cube[0], :, :]
            #     back = source[:, -self.cube[0]:, :, :]
            #
            #     front = front.unfold(2, self.cube[1], self.stride[2]).unfold(3, self.cube[1], self.stride[2])
            #     back = back.unfold(2, self.cube[1], self.stride[2]).unfold(3, self.cube[1], self.stride[2])
            #
            #     num_blocks = front.shape[2:4]
            #
            #     front = front.permute(2, 3, 0, 1, 4, 5).contiguous()
            #     front = front.view(-1, 3, self.cube[0], self.cube[1], self.cube[2])
            #     back = back.permute(2, 3, 0, 1, 4, 5).contiguous()
            #     back = back.view(-1, 3, self.cube[0], self.cube[1], self.cube[2])
            #
            #     source = torch.cat((front, back), 0)

            if itr == 0:
                self.data_tensor = source
            else:
                self.data_tensor = torch.cat((self.data_tensor, source), 0)

            self.info = {
                'x_pad': x_pad,
                'y_pad': y_pad,
                'z_pad': z_pad,
                'stride': self.stride,
                'orig_shape': shape,
                'pad_shape': pad_shape,
                'num_blocks': num_blocks
            }

    def __getitem__(self, item):

        return self.data_tensor[item]

    def __len__(self):
        return len(self.data_tensor)


# class InferDataset(data.Dataset):
#
#     def __init__(self, target_volume_list, label_volume_list, cube_size, seed):
#
#         self.source = target_volume_list
#         self.target = label_volume_list
#         self.random = np.random.RandomState(seed)
#         self.cubeSize = cube_size
#         self.centerCrop = T.CenterCrop((640, 896))
#
#     def __getitem__(self, item):
#
#         # Because we are inffering, we know we are more interested in the center of the volume
#         source = self.source[item]
#         target = self.target[item]
#
#         source = self.centerCrop(source)[:, :, :, 0:64] # only take the front
#         target = self.centerCrop(target)[:, :, :, 0:64]
#
#         source = source.unfold(1, 128, 128).unfold(2, 128, 128).unfold(3, 64, 64)
#         target = target.unfold(1, 128, 128).unfold(2, 128, 128).unfold(3, 64, 64)
#
#         source.view(-1, 128, 128, 64)
#         target.view(-1, 128, 128, 64)
#
#         return source, target
#
#     def __len__(self):
#         return len(self.source)
