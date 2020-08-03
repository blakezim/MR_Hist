import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from skimage import color
from PIL.Image import BILINEAR


class TrainDataset(data.Dataset):

    def __init__(self, inputs, labels, length, im_size):
        super(TrainDataset, self).__init__()

        self.inputs = inputs
        self.labels = labels
        self.length = length
        self.size = im_size

    def spatial_transform(self, input1, label1):

        vflip = torch.rand(1).item() > 0.5
        hflip = torch.rand(1).item() > 0.5
        deg = torch.LongTensor(1).random_(-20, 20).item()
        scale = torch.FloatTensor(1).uniform_(0.8, 1.2)

        image_list = [input1, label1]
        for i, p in enumerate(image_list):
            dtype = p.dtype

            p_min = p.min()
            p_max = p.max()

            p = (p - p_min) / ((p_max - p_min) + 0.001)

            p = TF.to_pil_image(p.float())
            if vflip:
                p = TF.vflip(p)
            if hflip:
                p = TF.hflip(p)

            p = TF.affine(p, deg, scale=scale, translate=(0, 0), shear=0, resample=BILINEAR)
            p = TF.to_tensor(p).squeeze()

            p = (p * ((p_max - p_min) + 0.001)) + p_min

            if dtype == torch.int64:
                p = p.round()
            p = p.to(dtype=dtype)

            image_list[i] = p.clone()

        input1, mask = image_list

        return input1, mask

    def __getitem__(self, item):
        # # #
        # import matplotlib
        # matplotlib.use('qt5agg')
        # import matplotlib.pyplot as plt
        # plt.ion()
        idx = item % self.length

        input1 = self.inputs[idx]
        label1 = self.labels[idx]

        if any([x < self.size for x in list(input1.shape[1:3])]):
            size_diff = torch.tensor([input1.shape[0], self.size, self.size]) - torch.tensor(input1.shape)
            size_diff[size_diff < 0.0] = 0.0
            pad = (size_diff[2] // 2, size_diff[2] // 2 + size_diff[2] % 2,
                   size_diff[1] // 2, size_diff[1] // 2 + size_diff[1] % 2)
            input1 = F.pad(input1, pad)
            label1 = F.pad(label1, pad)

        temp = torch.zeros((self.size, self.size))

        while temp.sum() == 0.0:
            if input1.shape[2] == self.size:
                x = 0
            else:
                x = torch.randint(0, (input1.shape[2] - self.size), (1,)).item()

            if input1.shape[1] == self.size:
                y = 0
            else:
                y = torch.randint(0, (input1.shape[1] - self.size), (1,)).item()
            temp = label1[y: y + self.size, x: x + self.size]

        input1 = input1[:, y: y + self.size, x: x + self.size]
        label1 = label1[y: y + self.size, x: x + self.size]

        # Spatially trasform the source and target
        input1, mask = self.spatial_transform(input1, label1.long())
        # mask = torch.round(mask)

        input_hsv = torch.from_numpy(color.rgb2hsv(input1.permute(1, 2, 0).numpy())).permute(2, 0, 1)

        inputs = torch.cat([input1, input_hsv], dim=0)

        return inputs.float(), mask.long()

    def __len__(self):
        return self.length


class EvalDataset(data.Dataset):
    def __init__(self, inputs, labels, length, im_size):
        super(EvalDataset, self).__init__()

        self.inputs = inputs
        self.labels = labels
        self.length = length
        self.size = im_size

    def __getitem__(self, item):

        input1 = self.inputs[item]
        label1 = self.labels[item]

        if any([x < self.size for x in list(input1.shape[1:3])]):
            size_diff = torch.tensor([input1.shape[0], self.size, self.size]) - torch.tensor(input1.shape)
            size_diff[size_diff < 0.0] = 0.0
            pad = (size_diff[2] // 2, size_diff[2] // 2 + size_diff[2] % 2,
                   size_diff[1] // 2, size_diff[1] // 2 + size_diff[1] % 2)
            input1 = F.pad(input1, pad)

        temp = torch.zeros((self.size, self.size))

        while temp.sum() == 0.0:
            x = torch.randint(0, (input1.shape[2] - self.size), (1,)).item()
            y = torch.randint(0, (input1.shape[1] - self.size), (1,)).item()
            temp = label1[y: y + self.size, x: x + self.size]

        input1 = input1[:, y: y + self.size, x: x + self.size]
        label1 = label1[y: y + self.size, x: x + self.size]

        input_hsv = torch.from_numpy(color.rgb2hsv(input1.permute(1, 2, 0).numpy())).permute(2, 0, 1)

        inputs = torch.cat([input1, input_hsv], dim=0)

        return inputs.float(), label1.long()

    def __len__(self):
        return self.length()
