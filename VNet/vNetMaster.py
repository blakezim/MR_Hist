import os
import sys
sys.path.append('./')
import glob
import time
import torch
import torch.nn as nn
import subprocess as sp
import torch.optim as optim
import matplotlib

from vNetModel import vnet_model
from collections import OrderedDict

matplotlib.use('Qt5Agg')


from types import SimpleNamespace
from losses import dice_loss
from dataset import TrainDataset
from learningrate import CyclicLR
from data_prep import generate_input_block
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


def _get_branch(opt):
    p = sp.Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], shell=False, stdout=sp.PIPE)
    branch, _ = p.communicate()
    branch = branch.decode('utf-8').split()

    p = sp.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=sp.PIPE)
    hash, _ = p.communicate()
    hash = hash.decode('utf-8').split()

    opt.git_branch = branch
    opt.git_hash = hash


def _check_branch(opt, params):
    """ When performing eval, check th git branch and commit that was used to generate the .pt file"""

    # Check the current branch and hash
    _get_branch(opt)

    if params.git_branch != opt.git_branch or params.git_hash != opt.git_hash:
        msg = 'You are not on the right branch or commit. Please run the following in the repository: \n'
        msg += f'git checkout {params.git_branch}\n'
        msg += f'git revert {params.git_hash}'
        sys.exit(msg)


def get_loader(source, target, state, opt, stride=None):
    num_blocks = len(source)
    rep_factor = 40

    # Need to redo this so that only unseen blocks are in the testing

    data_length = num_blocks * rep_factor

    if state == 'train':
        dataset = TrainDataset(source, target, opt.cube, int(data_length * 0.9))
        sampler = SubsetRandomSampler(range(0, int(data_length * 0.9)))
        return DataLoader(dataset, opt.trainBatchSize, sampler=sampler, num_workers=opt.threads)

    elif state == 'infer':
        dataset = TrainDataset(source, target, opt.cube, data_length - int(data_length * 0.9))
        sampler = SequentialSampler(range(int(data_length * 0.9), data_length))
        return DataLoader(dataset, opt.inferBatchSize, sampler=sampler, num_workers=opt.threads)


def save_volume(vol, opt):
    import CAMP.Core as core
    import CAMP.FileIO as io
    # Bring the volume to the CPU
    vol = vol.to('cpu')
    out_path = f'{opt.hd_root}/{opt.rabbit}/blockface/{opt.block}/volumes/raw/'

    mhd_file = f'{out_path}/surface_volume.mhd'
    mhd_dict = read_mhd_header(mhd_file)

    data_files = f'predictions/IMG_%03d_prediction.raw 0 {vol.shape[0]} 1'
    mhd_dict['ElementDataFile'] = data_files
    mhd_dict['ElementNumberOfChannels'] = 1

    out_path = f'{out_path}/predictions/'
    try:
        os.stat(out_path)
    except OSError:
        os.makedirs(out_path)

    origin = [float(x) for x in mhd_dict['Offset'].split(' ')][0:2]
    spacing = [float(x) for x in mhd_dict['ElementSpacing'].split(' ')][0:2]

    # Loop over the firs dimension
    for s in range(0, vol.shape[0]):

        v_slice = core.StructuredGrid(
            vol.shape[1:],
            tensor=vol[s, :, :].unsqueeze(0),
            spacing=spacing,
            origin=origin,
            device='cpu',
            dtype=torch.float32,
            channels=1
        )

        # Generate the name for the image
        out_name = f'{out_path}IMG_{s:03d}_prediction.mhd'
        io.SaveITKFile(v_slice, out_name)

    write_mhd_header(f'{out_path}../prediction_volume.mhd', mhd_dict)


def read_mhd_header(filename):

    with open(filename, 'r') as in_mhd:
        long_string = in_mhd.read()

    short_strings = long_string.split('\n')
    key_list = [x.split(' = ')[0] for x in short_strings[:-1]]
    value_list = [x.split(' = ')[1] for x in short_strings[:-1]]
    a = OrderedDict(zip(key_list, value_list))

    return a


def write_mhd_header(filename, dictionary):
    long_string = '\n'.join(['{0} = {1}'.format(k, v) for k, v in dictionary.items()])
    with open(filename, 'w+') as out:
        out.write(long_string)


def train(opt):

    # Add the git information to the opt
    _get_branch(opt)
    timestr = time.strftime("%Y-%m-%d-%H%M%S")
    writer = SummaryWriter('/usr/sci/scratch/blakez/blockface_data/runs/{0}/'.format(timestr))
    writer.add_text('Parameters', opt.__str__())
    dataDir = '/usr/sci/scratch/blakez/blockface_data/'

    torch.manual_seed(opt.seed)

    try:
        os.stat('/usr/sci/scratch/blakez/blockface_data/output/{0}/'.format(timestr))
    except OSError:
        os.makedirs('/usr/sci/scratch/blakez/blockface_data/output/{0}/'.format(timestr))

    # Load the data
    print('===> Loading Data')
    sourceTensor = []
    labelTensor = []

    for file in sorted(glob.glob(dataDir + 'inputs/*.pth')):
        sourceTensor.append(torch.load(file))
    for file in sorted(glob.glob(dataDir + 'labels/*.pth')):
        labelTensor.append(torch.load(file))

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('===> Generating Datasets')
    training_data_loader = get_loader(sourceTensor, labelTensor, 'train', opt)
    infering_data_loader = get_loader(sourceTensor, labelTensor, 'infer', opt)

    model = vnet_model.VNet()
    crit = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=1e-6, momentum=0.9, nesterov=True)

    if cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()

    if opt.scheduler:
        scheduler = CyclicLR(optimizer, step_size=500, max_lr=10*opt.lr, base_lr=opt.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    if opt.resume:
        params = torch.load(opt.ckpt)
        model.load_state_dict(params['state_dict'])
        optimizer.load_state_dict(params['optimizer'])
        # if params['scheduler']:
        #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    def checkpoint(state):
        path = '/usr/sci/scratch/blakez/blockface_data/output/{0}/epoch_{1}_model.pth'.format(timestr, epoch)
        torch.save(state, path)
        print("===> Checkpoint saved for Epoch {}".format(epoch))

    def learn(epoch):
        model.train()
        epoch_loss = 0

        for iteration, batch in enumerate(training_data_loader, 1):
            scheduler.batch_step()
            input, target = Variable(batch[0]), Variable(batch[1])

            if cuda:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            pred = model(input)

            loss = crit(pred.squeeze(), target.float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Batch/Learning Rate', scheduler.get_lr()[-1],
                              (iteration + (len(training_data_loader) * (epoch - 1))))
            writer.add_scalar('Batch/BCE Loss', loss.item(),
                              (iteration + (len(training_data_loader) * (epoch - 1))))
            print(
                "=> Done with {} / {}  Batch Loss: {:.6f}".format(iteration, len(training_data_loader), loss.item()))
        writer.add_scalar('Epoch/BCE Loss', epoch_loss / len(training_data_loader), epoch)
        print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))

    def infer(epoch):
        sigmoid = nn.Sigmoid()
        val_loss = 0
        dice = 0
        model.eval()
        print('===> Evaluating Model')
        with torch.no_grad():
            for iteration, batch in enumerate(infering_data_loader, 1):
                input, target = Variable(batch[0]), Variable(batch[1])

                if cuda:
                    input = input.cuda()
                    target = target.cuda()

                pred = model(input)

                loss = crit(pred.squeeze(1), target.float()).item()
                val_loss += loss

                # Turn the prediction into probabilities
                pred = sigmoid(pred)
                mask = (pred[:, 0, :, :, :] > 0.5).float()

                dice += dice_loss(mask, target.float()).item()

                if iteration == 1:
                    non_zero = [i for i, x in enumerate(target, 0) if x.sum() != 0]
                    if not non_zero:
                        writer.add_image('Test/Input', input[0, 0:3, 32, :, :].squeeze(), epoch)
                        writer.add_image('Test/Prediction', pred[0, 0, 32, :, :].repeat(3, 1, 1), epoch)
                        writer.add_image('Test/Segmentation', mask[0, 32, :, :].repeat(3, 1, 1), epoch)
                        writer.add_image('Test/Ground Truth', target[0, 32, :, :].repeat(3, 1, 1), epoch)
                    else:
                        index = non_zero[0]
                        writer.add_image('Test/Input', input[index, 0:3, 32, :, :].squeeze(), epoch)
                        writer.add_image('Test/Prediction', pred[index, 0, 32, :, :].repeat(3, 1, 1), epoch)
                        writer.add_image('Test/Segmentation', mask[index, 32, :, :].repeat(3, 1, 1), epoch)
                        writer.add_image('Test/Ground Truth', target[index, 32, :, :].repeat(3, 1, 1), epoch)
                print("=> Done with {} / {}  Batch Loss: {:.6f}".format(iteration, len(infering_data_loader), loss))

        writer.add_scalar('test/Average Loss', val_loss / len(infering_data_loader), epoch)
        writer.add_scalar('test/Average DICE', dice / len(infering_data_loader), epoch)

        print("===> Avg. DICE: {:.6f}".format(dice / len(infering_data_loader)))

    print("===> Beginning Training")

    if opt.resume:
        epochs = range(params['epoch'], opt.nEpochs)
    else:
        epochs = range(1, opt.nEpochs)

    for epoch in epochs:
        print("===> Learning Rate = {}".format(optimizer.param_groups[0]['lr']))
        learn(epoch)
        if epoch % 10 == 0:
            checkpoint({
                'epoch': epoch,
                'scheduler': opt.scheduler,
                'git_branch': opt.git_branch,
                'git_hash': opt.git_hash,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()})
        infer(epoch)
        # if opt.scheduler:
        #     scheduler.step(epoch)


def eval(opt):
    # Load the data
    print('===> Loading Data ... ', end='')
    # Need to get the original size of the data
    input_vol, orig_shape = generate_input_block(opt.block, opt.rabbit, opt=opt)
    print('Done')

    print('===> Loading Model ... ', end='')
    model = vnet_model.VNet()

    if opt.cuda:
        input_vol = input_vol.cuda()
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        params = torch.load(opt.ckpt)
        model.load_state_dict(params['state_dict'])
    else:
        params = torch.load(opt.ckpt, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])

    sigmoid = nn.Sigmoid()
    model.eval()
    print('Done')

    print('===> Evaluating ... ', end='')
    with torch.no_grad():

        pred = model(input_vol.unsqueeze(0))
        torch.cuda.empty_cache()
        pred = sigmoid(pred)

    print('Done')

    print('===> Resizing and Saving ... ', end='')
    pred = torch.nn.functional.interpolate(pred, size=orig_shape[-3:],
                                           mode='trilinear', align_corners=True)
    with torch.no_grad():
        # For fun, try a laplacian filter to sharpen the edges
        kernel = [[[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]],
                  [[0, 1, 0],
                   [1, -6, 1],
                   [0, 1, 0]],
                  [[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]]
                  ]
        l_filt = nn.Conv3d(1, 1, (3, 3, 3), stride=1, padding=1, bias=False)
        l_filt.weight[0] = torch.tensor(kernel).float()
        l_filt = l_filt.to('cuda')

        pred = (pred - l_filt(pred)).squeeze()

    save_volume(pred, opt)
    print('Done')

    print('===> Evaluation Complete')


if __name__ == '__main__':
    trainOpt = {'trainBatchSize': 10,
                'inferBatchSize': 10,
                'nEpochs': 500,
                'lr': 0.0007,
                'cuda': True,
                'threads': 4,
                'cube': [64, 128, 128],
                'resume': False,
                'scheduler': True,
                'seed': 5946,
                'ckpt': None
                }

    evalOpt = {'evalBatchSize': 4,
               'rabbit': '18_062',
               'block': 'block07',
               'cuda': True,
               'threads': 0,
               'hd_root': '/hdscratch/ucair/',
               'cube': [64, 128, 128],
               'ckpt': '/scratch/ucair/blockface/output/2019-04-25-160303/epoch_70_model.pth',
               }

    evalOpt = SimpleNamespace(**evalOpt)
    trainOpt = SimpleNamespace(**trainOpt)

    train(trainOpt)
    # eval(evalOpt)
    print('All Done')
