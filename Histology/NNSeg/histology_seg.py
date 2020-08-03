import os
import sys
sys.path.append('/home/sci/blakez/ucair/MR_Hist/')
# import glob
import math
import time
import torch
import glob
# import tools
# import model as densenet
# import numpy as np
# import losses
import torch.nn as nn
import subprocess as sp
import torch.optim as optim
import argparse
import numpy as np

from Histology.NNSeg.dataset import TrainDataset, EvalDataset
from Histology.NNSeg.models.UNet import UNet
# import SimpleITK as sitk
# import torch.nn.functional as F

# import matplotlib
# from models import FC3b3, init_weights

# from VNet.vNetModel import vnet_model
# from collections import OrderedDict

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from scipy.io import loadmat, savemat
from types import SimpleNamespace
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from scipy.ndimage.morphology import *
# from math import floor
# from VNet.losses import dice_loss
# from dataset import TrainDataset, EvalDataset
# from learningrate import CyclicLR
# from VNet.data_prep import generate_input_block
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


parser = argparse.ArgumentParser(description='PyTorch Patch Based Super Deep Interpolation Example')

parser.add_argument('-d', '--data_dir', type=str, default='/usr/sci/scratch/blakez/microscopic_data/',
                    help='Path to data directory')
parser.add_argument('-o', '--out_dir', type=str, default='/usr/sci/scratch/blakez/microscopic_data/Output/',
                    help='Path to output')
parser.add_argument('--trainBatchSize', type=int, default=64, help='training batch size')
parser.add_argument('--inferBatchSize', type=int, default=64, help='cross validation batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=358, help='random seed to use. Default=358')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--im_size', type=int, default=256, help='number of threads for data loader to use')
parser.add_argument('-r', '--repeat_factor', type=int, default=2, help='repeat data factor')

opt = parser.parse_args()
print(opt)


class Sample:
    def __init__(self, sos1, sos2):

        self.ims1 = sos1.ims.clone()
        self.ims2 = sos2.ims.clone()
        if (sos1.mask - sos2.mask).sum().item() != 0:
            raise Exception('Masks are not the same.')

        if (sos1.label - sos2.label).sum().item() != 0:
            raise Exception('Labels are not the same.')

        if sos1.echo != sos2.echo:
            raise Exception('Echos are not the same.')

        if sos1.fa == sos2.fa:
            raise Exception('Flip angles are the same.')

        self.mask = sos1.mask
        self.label = sos1.label
        self.fa1 = sos1.fa
        self.fa2 = sos2.fa
        self.tr = sos1.tr
        self.echo = sos1.echo

    def get_fas(self):
        return tuple([self.fa1, self.fa2])


def _get_branch(opt):
    p = sp.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], shell=False, stdout=sp.PIPE)
    branch, _ = p.communicate()
    branch = branch.decode('utf-8').split()

    p = sp.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=sp.PIPE)
    hash, _ = p.communicate()
    hash = hash.decode('utf-8').split()

    opt.git_branch = branch
    opt.git_hash = hash


def _check_branch(opt, saved_dict, model):
    """ When performing eval, check th git branch and commit that was used to generate the .pt file"""

    # Check the current branch and hash
    _get_branch(opt)

    try:
        if opt.cuda:
            device = torch.device('cuda')
            model = model.to(device=device)
            model = nn.DataParallel(model)
            params = torch.load(f'{opt.model_dir}{opt.ckpt}')
            model.load_state_dict(params['state_dict'])
        else:
            device = torch.device('cpu')
            params = torch.load(f'{opt.model_dir}{opt.ckpt}', map_location='cpu')
            model.load_state_dict(params['state_dict'])
    except:
        raise Exception(f'The checkpoint {opt.ckpt} could not be loaded into the given model.')

    # if saved_dict.git_branch != opt.git_branch or saved_dict.git_hash != opt.git_hash:
    #     msg = 'The model loaded, but you are not on the same branch or commit.'
    #     msg += 'To check out the right branch, run the following in the repository: \n'
    #     msg += f'git checkout {params.git_branch[0]}\n'
    #     msg += f'git revert {params.git_hash[0]}'
    #     raise Warning(msg)

    return model, device


def add_figure(tensor, writer, title=None, text=None, label=None, epoch=0, min_max=None):

    import matplotlib.pyplot as plt

    font = {'color': 'white', 'size': 20}
    plt.figure()
    if min_max:
        plt.imshow(tensor.squeeze().cpu(), vmin=min_max[0], vmax=min_max[1])
    else:
        plt.imshow(tensor.squeeze().cpu())
    plt.colorbar()
    plt.axis('off')
    if title:
        plt.title(title)
    if text:
        if type(text) == list:
            for i, t in enumerate(text, 1):
                plt.text(5, i*25, t, fontdict=font)

    writer.add_figure(label, plt.gcf(), epoch)
    plt.close('all')


def get_loaders(opt):

    inputs = torch.load(f'{opt.dataDirectory}histology_inputs_two_label.pth')
    labels = torch.load(f'{opt.dataDirectory}histology_labels_two_label.pth')

    samp_split = int(np.floor(len(inputs) * 0.9))

    train_inputs = inputs[0:samp_split] * opt.repeat_factor
    train_lables = labels[0:samp_split] * opt.repeat_factor

    eval_inputs = inputs[samp_split:] * opt.repeat_factor
    eval_labels = labels[samp_split:] * opt.repeat_factor

    dataset = TrainDataset(train_inputs, train_lables, samp_split, opt.im_size)
    sampler = SubsetRandomSampler(range(0, len(train_inputs)))
    train_loader = DataLoader(dataset, opt.trainBatchSize, sampler=sampler, num_workers=opt.threads)

    infer_dataset = EvalDataset(eval_inputs, eval_labels, len(inputs) - samp_split, opt.im_size)
    infer_sampler = SequentialSampler(range(0, len(eval_inputs)))
    infer_loader = DataLoader(infer_dataset, opt.trainBatchSize, sampler=infer_sampler, num_workers=opt.threads)

    return train_loader, infer_loader


def learn(opt):

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.ion()

    def checkpoint(state, opt, epoch):
        path = f'{opt.outDirectory}/saves/{opt.timestr}/epoch_{epoch:05d}_model.pth'
        torch.save(state, path)
        print(f"===> Checkpoint saved for Epoch {epoch}")

    def train(epoch, scheduler):
        model.train()
        crit = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=0)
        b_losses = []

        for iteration, batch in enumerate(training_data_loader, 1):
            inputs, label = [batch[0].to(device=device), batch[1].to(device=device)]

            optimizer.zero_grad()
            pred = model(inputs)

            loss = crit(pred, label)
            loss.backward()

            b_loss = loss.item()
            b_losses.append(loss.item())
            optimizer.step()

            # If half way through the epoch, add a random sample to view
            if iteration == len(training_data_loader) // 2:
                with torch.no_grad():
                    im = len(inputs) // 2

                    input1 = inputs[im, 0:3].squeeze()
                    add_figure(input1.permute(1, 2, 0), writer, title='Input 1', label='Train/Hist', epoch=epoch,
                               text=None)

                    # Add the prediction
                    pred_im = pred[im].squeeze()
                    pred_im = torch.max(softmax(pred_im), dim=0)[1]
                    label_im = label[im].squeeze()
                    add_figure(pred_im, writer, title='Predicted Mask', label='Train/Pred Mask', epoch=epoch,
                               min_max=[0.0, 3.0])
                    # Add the stir
                    add_figure(label_im, writer, title='Real Mask', label='Train/Real Mask', epoch=epoch,
                               min_max=[0.0, 3.0])

            for param_group in optimizer.param_groups:
                clr = param_group['lr']
            writer.add_scalar('Batch/Learning Rate', clr, (iteration + (len(training_data_loader) * (epoch - 1))))
            writer.add_scalar('Batch/Avg. MSE Loss', b_loss, (iteration + (len(training_data_loader) * (epoch - 1))))
            print("=> Done with {} / {}  Batch Loss: {:.6f}".format(iteration, len(training_data_loader), b_loss))

        e_loss = torch.tensor(b_losses).sum() / len(training_data_loader)
        writer.add_scalar('Epoch/Avg. MSE Loss', e_loss, epoch)
        print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, e_loss))
        scheduler.step(e_loss / len(training_data_loader))

    def infer(epoch):
        crit = nn.CrossEntropyLoss()
        b_losses = []
        softmax = nn.Softmax(dim=0)
        model.eval()
        print('===> Evaluating Model')
        with torch.no_grad():
            for iteration, batch in enumerate(testing_data_loader, 1):
                inputs, label = [batch[0].to(device=device), batch[1].to(device=device)]

                pred = model(inputs)
                loss = crit(pred, label)

                b_loss = loss.item()
                b_losses.append(loss.item())

                if iteration == len(testing_data_loader) // 2:
                    im = len(inputs) // 2
                    print(im)
                    print(iteration)
                    pred_im = pred[im].squeeze()
                    pred_im = torch.max(softmax(pred_im), dim=0)[1]
                    label_im = label[im].squeeze()
                    input1 = inputs[im, 0:3].squeeze()

                    # Add the input images - they are not going to change
                    add_figure(input1.permute(1, 2, 0), writer, title='Input 1', label='Infer/Hist', epoch=epoch,
                               text=None)

                    # Add the stir
                    add_figure(label_im, writer, title='Real Mask', label='Infer/Real Mask', epoch=epoch,
                               min_max=[0.0, 3.0])

                    # Add the prediction
                    add_figure(pred_im, writer, title='Predicted Mask', label='Infer/Pred Mask', epoch=epoch,
                               min_max=[0.0, 3.0])

                print(f"=> Done with {iteration} / {len(testing_data_loader)}  Batch Loss: {b_loss:.6f}")

            e_loss = torch.tensor(b_losses).sum() / len(testing_data_loader)
            writer.add_scalar('Infer/Avg. MSE Loss', e_loss, epoch)
            print(f"===> Avg. MSE Loss: {e_loss:.6f}")

    # Add the git information to the opt
    _get_branch(opt)
    timestr = time.strftime("%Y-%m-%d-%H%M%S")
    opt.timestr = timestr
    writer = SummaryWriter(f'{opt.outDirectory}/runs/{timestr}')
    writer.add_text('Parameters', opt.__str__())

    # Seed anything random to be generated
    torch.manual_seed(opt.seed)

    try:
        os.stat(f'{opt.outDirectory}/saves/{timestr}/')
    except OSError:
        os.makedirs(f'{opt.outDirectory}/saves/{timestr}/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    print('===> Generating Datasets ... ', end='')
    training_data_loader, testing_data_loader = get_loaders(opt)
    print(' done')

    model = UNet(6, 3)
    model = model.to(device)
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True, factor=0.5,
                                                     threshold=5e-3, cooldown=75, min_lr=1e-5)

    print("===> Beginning Training")

    epochs = range(1, opt.nEpochs + 1)

    for epoch in epochs:
        print("===> Learning Rate = {}".format(optimizer.param_groups[0]['lr']))
        train(epoch, scheduler)
        if epoch % 10 == 0:
            checkpoint({
                'epoch': epoch,
                'scheduler': opt.scheduler,
                'git_branch': opt.git_branch,
                'git_hash': opt.git_hash,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, opt, epoch)
        infer(epoch)


def eval(opt):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()
    from matplotlib import cm
    from scipy.ndimage.morphology import binary_erosion, binary_fill_holes

    print('===> Loading Data ... ', end='')

    infer_samples = torch.load(f'{opt.dataDirectory}/infer_objects.pt')

    infer_dataset = EvalDataset(infer_samples, len(infer_samples), opt.im_size)
    infer_sampler = SequentialSampler(range(0, len(infer_samples)))
    infer_loader = DataLoader(infer_dataset, opt.inferBatchSize, sampler=infer_sampler, num_workers=opt.threads)

    print('done')

    print('===> Loading Model ... ', end='')
    if not opt.ckpt:
        timestamps = sorted(glob.glob(f'{opt.model_dir}/*'))
        if not timestamps:
            raise Exception(f'No save directories found in {opt.model_dir}')
        lasttime = timestamps[-1].split('/')[-1]
        models = sorted(glob.glob(f'{opt.model_dir}/{lasttime}/*'))
        if not models:
            raise Exception(f'No models found in the last run ({opt.model_dir}{lasttime}/')
        model_file = models[-1].split('/')[-1]
        opt.ckpt = f'{lasttime}/{model_file}'

    model = unet_model.UNet(24, 1)
    # model = model.to(device)
    # model = nn.DataParallel(model)

    saved_dict = SimpleNamespace(**torch.load(f'{opt.model_dir}{opt.ckpt}'))
    model, device = _check_branch(opt, saved_dict, model)
    print('done')

    model.eval()
    crit = nn.L1Loss()
    e_loss = 0.0
    preds = []
    inputs_ims = []
    labels = []
    masks = []
    t1s = []

    print('===> Evaluating Model')
    with torch.no_grad():
        for iteration, batch in enumerate(infer_loader, 1):
            inputs, t1, mask, label = [batch[0].to(device=device), batch[1].to(device=device),
                                       batch[2].to(device=device), batch[3].to(device=device)]

            pred = model(inputs).squeeze()
            preds.append(pred.clone())
            inputs_ims.append(inputs.clone())
            labels.append(label.clone())
            masks.append(mask.clone())
            t1s.append(t1.clone())

            loss = crit(pred[mask], label[mask])
            e_loss += loss.item()
            b_loss = loss.item()

            print(f"=> Done with {iteration} / {len(infer_loader)}  Batch Loss: {b_loss:.6f}")

        print(f"===> Avg. MSE Loss: {e_loss / len(infer_loader):.6f}")

    pred_list = list(torch.cat(preds).cpu().split(1, dim=0))
    mask_list = list(torch.cat(masks).cpu().split(1, dim=0))
    errode_list = list(torch.cat(masks).cpu().split(1, dim=0))
    label_list = list(torch.cat(labels).cpu().split(1, dim=0))
    input_list = list(torch.cat(inputs_ims).cpu().split(1, dim=0))
    t1_list = list(torch.cat(t1s).cpu().split(1, dim=0))

    for i, mask in enumerate(errode_list):
        errode = binary_erosion(mask.squeeze(), iterations=2)
        fill = binary_fill_holes(errode)
        mask_list[i] = torch.tensor(mask * fill).unsqueeze(0).bool()

    fig_dir = f'./Output/figures/{opt.ckpt.split("/")[0]}/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    save_fig = False
    import scipy

    losses = []
    indexes = []

    for im in range(0, len(pred_list)):
        loss = crit(label_list[im].squeeze()[errode_list[im].squeeze()].unsqueeze(0),
                    pred_list[im].squeeze()[errode_list[im].squeeze()].unsqueeze(0))
        losses.append(loss)
        indexes.append(im)

    rvals = []
    rinds = []
    for im in range(0, len(pred_list)):
        rval = scipy.stats.linregress(label_list[im].squeeze()[errode_list[im].squeeze()],
                                      pred_list[im].squeeze()[errode_list[im].squeeze()])
        rvals.append(rval.rvalue)
        rinds.append(im)

    losses = torch.tensor(losses)
    indexes = torch.tensor(indexes)
    rvals = torch.tensor(rvals)
    rinds = torch.tensor(rinds)

    # slice_num = indexes[torch.min(losses, dim=0)[-1].item()]

    slice_num = rinds[torch.max(rvals, dim=0)[-1].item()]

    label_im = label_list[slice_num].squeeze()
    pred_im = pred_list[slice_num].squeeze()
    sos_im = t1_list[slice_num].squeeze()
    mask_im = mask_list[slice_num].squeeze()
    errode_im = errode_list[slice_num].squeeze()

    # erode the mask

    high_rval_inds = rinds[rvals > 0.9]

    fa1s = []
    fa2s = []
    for ind in high_rval_inds:
        fa1s.append(infer_samples[ind].fa1)
        fa2s.append(infer_samples[ind].fa2)
        print(f' FA1: {fa1s[-1]:.02f}   FA2: {fa2s[-1]:.02f}   R: {rvals[ind]:.02f}')

    sos_loss = crit(sos_im[mask_im].unsqueeze(0), label_im[mask_im].unsqueeze(0))
    pred_loss = crit(pred_im[mask_im].unsqueeze(0), label_im[mask_im].unsqueeze(0))

    ts = 20

    ET = torch.exp(-infer_samples[ts].tr / (label_list[ts] + 0.001))
    fa1 = infer_samples[ts].get_fas()[0] * (math.pi / 180.0)
    fa2 = infer_samples[ts].get_fas()[1] * (math.pi / 180.0)

    fa1 = infer_samples[slice_num].fa1
    fa2 = infer_samples[slice_num].fa2

    # Change the colormap background
    my_plasma = cm.get_cmap('plasma', 1024)
    my_plasma.colors[0] = np.array([0, 0, 0, 0])

    my_jet = cm.get_cmap('viridis', 1024)
    my_jet.colors[0] = np.array([0, 0, 0, 0])

    plt.close('all')

    def abline(slope, intercept):
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, 'k--')

    plt.figure()
    plt.imshow(pred_im * mask_im, vmin=0.0, vmax=1.0, cmap=my_plasma, )
    plt.title(f'NN T1 (L1: {pred_loss:.02f})')
    plt.colorbar()
    plt.axis('off')
    if save_fig:
        plt.savefig(f'{fig_dir}/NN_T1.png', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(label_im * mask_im, vmin=0.0, vmax=1.0, cmap=my_plasma, )
    plt.title('STIR T1')
    plt.colorbar()
    plt.axis('off')
    if save_fig:
        plt.savefig(f'{fig_dir}/STIR_T1.png', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.imshow(sos_im * mask_im, vmin=0.0, vmax=1.0, cmap=my_plasma, )
    plt.title(f'SoS T1 (No B1 Map) (L1: {sos_loss:.02f})')
    plt.colorbar()
    plt.axis('off')
    if save_fig:
        plt.savefig(f'{fig_dir}/SoS_T1.png', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.scatter(label_im[mask_im], sos_im[mask_im], cmap=my_plasma, c=label_im[mask_im], s=20)
    plt.gca().set_ylim(0.0, 2.0)
    plt.gca().set_xlim(0.0, 2.0)
    plt.xlabel('STIR T1 Values')
    plt.ylabel('SoS T1 Values (No B1 Map)')
    abline(1.0, 0.0)
    if save_fig:
        plt.savefig(f'{fig_dir}/SoS_Comp.png', dpi=600, bbox_inches='tight', pad_inches=0)

    plt.figure()
    plt.scatter(label_im[mask_im], pred_im[mask_im], cmap=my_plasma, c=label_im[mask_im], s=20)
    plt.gca().set_ylim(0.0, 2.0)
    plt.gca().set_xlim(0.0, 2.0)
    plt.xlabel('STIR T1 Values')
    plt.ylabel('NN T1 Values')
    abline(1.0, 0.0)
    if save_fig:
        plt.savefig(f'{fig_dir}/NN_Comp.png', dpi=600, bbox_inches='tight', pad_inches=0)


    # plt.figure()
    # plt.imshow(input_list[slice_num].squeeze()[0] * mask_im, vmin=0.0, vmax=3.0, cmap=my_jet)
    # plt.title(f'Input 1 (FA: {fa1:.02f})')
    # plt.colorbar()
    # plt.axis('off')
    # if save_fig:
    #     plt.savefig(f'{fig_dir}/Input1.png', dpi=600, bbox_inches='tight', pad_inches=0)
    #
    # plt.figure()
    # plt.imshow(input_list[slice_num].squeeze()[1] * mask_im, vmin=0.0, vmax=3.0, cmap=my_jet)
    # plt.title(f'Input 2 (FA: {fa2:.02f})')
    # plt.colorbar()
    # plt.axis('off')
    # if save_fig:
    #     plt.savefig(f'{fig_dir}/Input2.png', dpi=600, bbox_inches='tight', pad_inches=0)


    # Look at some inputs and predictions
    ims = [10, 20, 30, 40, 50]
    preds = torch.cat(preds, dim=0)
    inputs_ims = torch.cat(inputs_ims, dim=0)
    labels = torch.cat(labels, dim=0)
    masks = torch.cat(masks, dim=0)

    for im in ims:
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
        im1 = axs[0].imshow((inputs_ims[im, 0, :, :]*masks[im, :, :]).cpu(), cmap=my_jet, aspect=1)
        fig.colorbar(im1, ax=axs[0], shrink=0.5)
        axs[0].axis('off')
        im2 = axs[1].imshow((inputs_ims[im, 1, :, :]*masks[im, :, :]).cpu(), cmap=my_jet, aspect=1)
        fig.colorbar(im2, ax=axs[1], shrink=0.5)
        axs[1].axis('off')
        im3 = axs[2].imshow((preds[im, :, :]*masks[im, :, :]).cpu(), cmap=my_plasma, aspect=1, vmin=0.0, vmax=1.0)
        fig.colorbar(im3, ax=axs[2], shrink=0.5)
        axs[2].axis('off')
        im4 = axs[3].imshow((labels[im, :, :]*masks[im, :, :]).cpu(), cmap=my_plasma, aspect=1, vmin=0.0, vmax=1.0)
        fig.colorbar(im4, ax=axs[3], shrink=0.5)
        axs[3].axis('off')

    pred_vol = torch.cat(preds, dim=0)
    pred_vol = (pred_vol * 4000.0) - 1000.0
    pred_vol[pred_vol < -1000.0] = -1000
    pred_vol[pred_vol > 3000.0] = 3000.0
    pred_vol = pred_vol.permute(1, 2, 0)

    s = 300
    save_fig = True
    fig_dir = f'./Output/figures/{opt.model_dir.split("/")[-2]}/'

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    #
    # in1_im = ute1[:, :, s].squeeze().cpu()
    # in2_im = ute2[:, :, s].squeeze().cpu()
    # ct_im = ct[:, :, s].squeeze().cpu()
    # pred_im = pred_vol[:, :, s].squeeze().cpu()
    #
    # plt.figure()
    # plt.imshow(in1_im, vmin=in1_im.min(), vmax=in1_im.max(), cmap='plasma')
    # plt.axis('off')
    # plt.title('UTE 1')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}ute1.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])
    #
    # plt.figure()
    # plt.imshow(in2_im, vmin=in1_im.min(), vmax=in1_im.max(), cmap='plasma')
    # plt.axis('off')
    # plt.title('UTE 2')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}ute2.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])
    #
    # plt.figure()
    # plt.imshow(pred_im, vmin=-1000.0, vmax=3000.0, cmap='gray')
    # plt.axis('off')
    # plt.title('Predicted CT')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}pred_ct.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])
    #
    # plt.figure()
    # plt.imshow(ct_im, vmin=-1000.0, vmax=3000.0, cmap='gray')
    # plt.axis('off')
    # plt.title('Real CT')
    # plt.colorbar()
    # if save_fig:
    #     plt.savefig(f'{fig_dir}real_ct.png', dpi=600, bbox_inches='tight', pad_inches=0,
    #                 transparaent=True, facecolor=[0, 0, 0, 0])


if __name__ == '__main__':
    trainOpt = {'trainBatchSize': opt.trainBatchSize,
                'inferBatchSize': opt.inferBatchSize,
                'dataDirectory': opt.data_dir,
                'outDirectory': opt.out_dir,
                'nEpochs': opt.nEpochs,
                'lr': opt.lr,
                'cuda': opt.cuda,
                'threads': opt.threads,
                'resume': False,
                'scheduler': True,
                'ckpt': None,
                'seed': opt.seed,
                'im_size': opt.im_size,
                'repeat_factor': opt.repeat_factor
                }

    evalOpt = {'inferBatchSize': 128,
               'dataDirectory': './Data/PreProcessedData/AP_CombEchos/',
               'model_dir': './Output/saves/',
               'outDirectory': './Output/Predictions/',
               'cuda': True,
               'threads': 0,
               'ckpt': '2020-06-07-233313/epoch_00100_model.pth',
               'crop': 128,
               'im_size': 256
               }

    evalOpt = SimpleNamespace(**evalOpt)
    trainOpt = SimpleNamespace(**trainOpt)

    learn(trainOpt)
    # eval(evalOpt)
    print('All Done')
