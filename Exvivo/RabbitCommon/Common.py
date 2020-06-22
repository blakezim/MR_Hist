import yaml
import torch
import numpy as np
from . import *
from .commonD2N import *
# import commonD2N as dc
# import PyCA.Core as ca
# import PyCA.Common as common
# import PyCACalebExtras.Common as cc
# import PyCACalebExtras.SetBackend
# plt = PyCACalebExtras.SetBackend.SetBackend()
import matplotlib.image as im
import scipy.ndimage.filters
# from AppUtils import Config

from CAMP import Core
from CAMP import FileIO as io
from CAMP import StructuredGridOperators
from CAMP import StructuredGridTools


def SolveRigidAffine(src_points, tar_points, grid=None):
    '''Takes correspondance points of a source volume and a target volume
       and returns the rigid only affine matrix formatted for use with PyCA
       apply affine functions.

       src_points = points chosen in the source (moving) volume
       tar_points = points chosen in the target (fixed) volume
       type = both sets of points are assumed to be N x Dim column numpy array

       This code is an implementation of the following technical notes:
       Sorkine-Hornung, Olga, and Michael Rabinovich. "Least-squares rigid motion using svd." Computing 1 (2017): 1.'''

    if grid is not None:
        if np.shape(src_points)[1] == 2:
            src_points = np.array(src_points) * grid.spacing().tolist()[0:2] + grid.origin().tolist()[0:2]
            tar_points = np.array(tar_points) * grid.spacing().tolist()[0:2] + grid.origin().tolist()[0:2]

        else:
            src_points = np.array(src_points) * grid.spacing().tolist() + grid.origin().tolist()
            tar_points = np.array(tar_points) * grid.spacing().tolist() + grid.origin().tolist()

    # Calculate the mean of each set of points
    src_mean = np.mean(src_points, 0)
    tar_mean = np.mean(tar_points, 0)

    # Subtract the mean so the points are centered at [0, 0, 0]
    src_zero = src_points - src_mean
    tar_zero = tar_points - tar_mean

    # Calculate the covariance matrix
    S = np.matmul(np.transpose(src_zero), tar_zero)

    # SVD of the covariance matrix
    [U, _, V] = np.linalg.svd(S)

    # Create the weights matrix and incorporate the determinant of the rotation matrix
    if np.shape(src_points)[1] == 2:
        W = np.eye(2)
        W[1, 1] = np.linalg.det(np.matmul(np.transpose(V), np.transpose(U)))

    else:
        W = np.eye(3)
        W[2, 2] = np.linalg.det(np.matmul(np.transpose(V), np.transpose(U)))

    # Caluclate the rotation matrix
    R = np.matmul(np.transpose(V), np.matmul(W, np.transpose(U)))

    # Calculate the translation from the rotated points
    rot_src_points = np.matmul(R, np.transpose(src_points))
    translation = tar_mean - np.mean(rot_src_points, 1)

    # Construct the affine matrix for use in PyCA
    if np.shape(src_points)[1] == 2:
        affine = np.zeros([3, 3])
        affine[0:2, 0:2] = R
        affine[0:2, 2] = translation
        affine[2, 2] = 1
    else:
        affine = np.zeros([4, 4])
        affine[0:3, 0:3] = R
        affine[0:3, 3] = translation
        affine[3, 3] = 1

    return affine


def SolveAffineGrid(source_image, input_affine, rot_point=None):
    """Takes a source volume and an affine matrix and solves for the necessary
       grid in the target space of the affine. Essentially calculates the bounding
       box of the source image after apply the affine transformation

       source_image = volume to be transformed by the affine (Image3D type)
       input_affine = the affine transformation (4x4)

       Returns the grid for the transformed source volume in real coordinates"""

    # Make sure we don't mess with the incoming affine
    affine = np.copy(input_affine)

    # Get parameters from the incoming source image
    in_sz = source_image.size.tolist()
    in_sp = source_image.spacing.tolist()
    in_or = source_image.origin.tolist()

    # Acount for 0 indexing
    in_sz = np.array(in_sz) - 1

    # Extract the pure rotation and ignore scaling to find the final size of the volume
    # U, s, V = np.linalg.svd(affine[0:3, 0:3])
    # rotMat = np.eye(4)
    # rotMat[0:3, 0:3] = np.dot(U, V)

    # Get the corners of the volume in index coordinates
    inputCorners = np.array([0, 0, 0, 1])
    inputCorners = np.vstack((inputCorners, np.array([in_sz[0], 0, 0, 1])))
    inputCorners = np.vstack((inputCorners, np.array([0, in_sz[1], 0, 1])))
    inputCorners = np.vstack((inputCorners, np.array([in_sz[0], in_sz[1], 0, 1])))

    # Account for the case that the source image is 3D
    if len(in_sz) == 3:
        inputCorners = np.vstack((inputCorners, np.array([0, 0, in_sz[2], 1])))
        inputCorners = np.vstack((inputCorners, np.array([in_sz[0], 0, in_sz[2], 1])))
        inputCorners = np.vstack((inputCorners, np.array([0, in_sz[1], in_sz[2], 1])))
        inputCorners = np.vstack((inputCorners, np.array([in_sz[0], in_sz[1], in_sz[2], 1])))

    # Define the index corners to find the final size of the transformed volume
    indexCorners = np.matrix(inputCorners)

    # Find the real corners of the input volume for finding the output origin
    realCorners = np.matrix(np.multiply(inputCorners, np.array(in_sp + [1])) + np.array(in_or + [0]))
    # Apply the transformations to the real and index corners
    # Have to subtract the mean here
    # Need to have the rot_point in both index and spacing so we can rotate both about that point
    # realMean[0, 0:3] -= realMean[0, 0:3] - np.array(rot_point)
    #  # This is so the translation is still included

    # indxMean[0, 0:3] -= indxMean[0, 0:3] - (np.array(rot_point) - np.array(in_or)) / np.array(in_sp)
    # For consistency sake - translation is going to be inreal coordiantes

    # Can't just do rotation can I? I need to adjust for the SCALE!!
    outRealCorners = np.matmul(affine, realCorners.transpose())
    outIndxCorners = np.matmul(affine, indexCorners.transpose())

    # Find the size in real and index coordinates of the output volume
    realSize = (np.max(outRealCorners, 1) - np.min(outRealCorners, 1))[0:3]
    indexSize = (np.max(outIndxCorners, 1) - np.min(outIndxCorners, 1))[0:3]

    # We can divide index size into real size to get the spacing of the real volume; Need to account for 2D zero
    # out_sp = np.squeeze(np.array(np.divide(realSize, indexSize, where=indexSize != 0)))
    # out_sp[out_sp == 0] = 1
    out_sp = np.abs(np.squeeze(np.array(affine * (np.matrix(in_sp + [0.0]).transpose())))).tolist()[0:3]

    out_sz = np.squeeze(np.array(np.ceil(realSize.transpose() / out_sp + 1).astype(int)))
    out_sz[out_sz == 0] = 1

    # Find the output origin by taking the min in each dimension of the real transformed corners
    out_or = np.squeeze(np.array(np.min(outRealCorners, 1)))[0:3]

    # Make the grid
    return Core.StructuredGrid(out_sz, spacing=out_sp, origin=out_or)


def UnionGrid(im1, im2):
    sz1 = im1.size().tolist()
    sp1 = im1.spacing().tolist()
    or1 = im1.origin().tolist()

    cor1 = np.array([0, 0, 0, 1])
    cor1 = np.vstack((cor1, np.array([sz1[0], 0, 0, 1])))
    cor1 = np.vstack((cor1, np.array([0, sz1[1], 0, 1])))
    cor1 = np.vstack((cor1, np.array([sz1[0], sz1[1], 0, 1])))

    # Account for the case that the source image is 3D
    if cc.Is3D(im1):
        cor1 = np.vstack((cor1, np.array([0, 0, sz1[2], 1])))
        cor1 = np.vstack((cor1, np.array([sz1[0], 0, sz1[2], 1])))
        cor1 = np.vstack((cor1, np.array([0, sz1[1], sz1[2], 1])))
        cor1 = np.vstack((cor1, np.array([sz1[0], sz1[1], sz1[2], 1])))

    sz2 = im2.size().tolist()
    sp2 = im2.spacing().tolist()
    or2 = im2.origin().tolist()

    cor2 = np.array([0, 0, 0, 1])
    cor2 = np.vstack((cor2, np.array([sz2[0], 0, 0, 1])))
    cor2 = np.vstack((cor2, np.array([0, sz2[1], 0, 1])))
    cor2 = np.vstack((cor2, np.array([sz2[0], sz2[1], 0, 1])))

    # Account for the case that the source image is 3D
    if cc.Is3D(im2):
        cor2 = np.vstack((cor2, np.array([0, 0, sz2[2], 1])))
        cor2 = np.vstack((cor2, np.array([sz2[0], 0, sz2[2], 1])))
        cor2 = np.vstack((cor2, np.array([0, sz2[1], sz2[2], 1])))
        cor2 = np.vstack((cor2, np.array([sz2[0], sz2[1], sz2[2], 1])))

    rCor1 = np.matrix(np.multiply(cor1, np.array(sp1 + [1])) + np.array(or1 + [0]))
    rCor2 = np.matrix(np.multiply(cor2, np.array(sp2 + [1])) + np.array(or2 + [0]))

    realSize = (np.max(np.max((rCor1, rCor2), 0), 0) - np.min(np.min((rCor1, rCor2), 0), 0))[0:3]

    out_sz = np.squeeze(
        np.array(np.ceil(np.divide(realSize, np.min((sp1, sp2), 0), where=np.min((sp1, sp2), 0) != 0))).astype(int))
    out_sp = np.min((sp1, sp2), 0)
    out_or = np.squeeze(np.array(np.min(np.min((rCor1, rCor2), 0), 0)))[0:3]

    return cc.MakeGrid(out_sz, out_sp, out_or)


def LoadDICOM(dicomDirectory, memType):
    '''Takes a directory that contains DICOM files and returns a PyCA Image3D

    dicomDirectory = Full path to the folder with the dicoms

    Returns an Image3D in the Reference Coordiante System (RCS)
    '''

    # Read the DICOM files in the directory
    dicoms = read_dicom_directory(dicomDirectory)

    # Sort the loaded dicoms
    sort_dcms = sort_dicoms(dicoms)

    # Extract the actual volume of pixels
    pixel_vol = get_volume_pixeldata(sort_dcms)
    pixel_vol = torch.tensor(pixel_vol.astype(np.int16))

    # Generate the affine from the dicom headers (THIS CODE WAS MODIFIED FROM dicom2nifti)
    affine, spacing, pp = create_affine(sort_dcms)

    # Convert the dicom volume to an Image3D
    rawDicom = Core.StructuredGrid(pixel_vol.shape, tensor=pixel_vol.unsqueeze(0), channels=1, spacing=[1.0, 1.0, 1.0],
                                   origin=[0.0, -1.0, 0.0])
    # rawDicom = common.ImFromNPArr(pixel_vol, memType)

    wcsTrans = np.eye(4)
    if pp == 'HFS':
        wcsTrans[0, 0] *= -1
        wcsTrans[1, 1] *= -1
    if pp == 'FFS':
        wcsTrans[1, 1] *= -1
        wcsTrans[2, 2] *= -1
    if pp == 'FFP':
        wcsTrans[0, 0] *= -1
        wcsTrans[2, 2] *= -1

    affine = torch.tensor(wcsTrans * affine, dtype=torch.float32)
    rcs_grid = SolveAffineGrid(rawDicom, affine)

    # world_grid = SolveAffineGrid(rcs_grid, wcsTrans)

    # Create the RCS Image3D and World Image3D with correct origin and spacing
    rcsImage = StructuredGridOperators.AffineTransform.Create(affine=affine)(rawDicom, rcs_grid)

    # Need to swap the first and third dimension for CAMP
    outImage = Core.StructuredGrid(
        size=rcsImage.size.tolist()[::-1],
        origin=rcsImage.origin.tolist()[::-1],
        spacing=rcsImage.spacing.tolist()[::-1],
        tensor=rcsImage.data.permute(0, 3, 2, 1),
        channels=1
    )

    return outImage


class PathSelector:

    def __init__(self, parent, paths, label):
        self.root = parent.Tk()
        self.root.title(label)
        self.root.geometry("1500x500")
        self.listbox = parent.Listbox(self.root, selectmode='multiple')
        [self.listbox.insert(i, path) for i, path in enumerate(paths)]

        self.listbox.pack(fill=parent.BOTH, expand=True, ipady=1)

        self.cvtBtn = parent.Button(self.root, text='Convert Files')
        self.cvtBtn.pack()
        self.cvtBtn.bind("<Button-1>", self.convert)

        parent.Button(self.root, text='Done', command=self.root.quit).pack()

        self.root.mainloop()

    def convert(self, event):
        self.files = []
        selections = self.listbox.curselection()
        for selection in selections:
            self.files.append(self.listbox.get(selection))

    def get_files(self):
        return self.files


def ReadList(filename):
    f = open(filename, 'r')
    readList = []
    for line in f:
        readList.append(line.strip())
    f.close()
    return readList


def WriteList(output_list, filename):
    f = open(filename, 'w')
    for i in output_list:
        f.write(str(i) + "\n")
    f.close()


def WriteConfig(spec, d, filename):
    """Write a Config Object to a YAML file"""
    with open(filename, 'w') as f:
        f.write(Config.ConfigToYAML(spec, d))
    f.close()


def FieldResampleWorld(f_out, f_in, bg=0):
    """Resamples fields with partial ID background strategy.
       Assumes the input field is in real coordinates.

       Returns: Resampled H Field in real coordinates."""

    # Create temp variable so that the orignal field is not modified
    temp = f_in.copy()
    # Change to index coordinates
    cc.HtoIndex(temp)
    # Convert the h field to a vector field
    ca.HtoV_I(temp)
    # Resample the field onto f_out grid
    ca.ResampleV(f_out, temp, bg=bg)
    # Convert back to h field
    ca.VtoH_I(f_out)
    # Convert the resampled field back to real coordinates
    cc.HtoReal(f_out)
    del temp


def MedianFilter_I(Im, kernel_size):
    """ In-place median filter for Image3D's """

    # Keep track of the memory type to return to
    memT = Im.memType()
    # Make sure the memeor is host
    Im.toType(ca.MEM_HOST)
    # Convert to a numpy array
    Im_np = Im.asnp()
    Im_np = scipy.ndimage.filters.median_filter(Im_np, kernel_size)

    ca.Copy(Im, common.ImFromNPArr(Im_np, ca.MEM_HOST, sp=Im.spacing().tolist(), orig=Im.origin().tolist()))

    # Return to original memory type
    Im.toType(memT)


def LandmarkPicker(imList):
    '''Allows the user to select landmark correspondences between any number of images. The images must be in a list and must be formatted as numpy arrays. '''

    class PointPicker(object):
        '''Image class for picking landmarks'''

        def __init__(self, X):
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.im = self.ax.imshow(X, cmap='gray')
            self.shape = np.shape(X)
            self.cords = []
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        def onclick(self, event):
            if event.button == 3:
                xidx = int(round(event.xdata))
                yidx = int(round(event.ydata))
                self.cords.append([xidx, yidx])
                self.plot()

        def plot(self):
            self.ax.scatter(self.cords[-1][0], self.cords[-1][1])
            self.ax.set_xlim([0, self.shape[1]])
            self.ax.set_ylim([self.shape[0], 0])
            plt.pause(0.001)

    plt.ion()
    pickerList = [PointPicker(im) for im in imList]
    plt.show()
    # while all([p.fig.get_visible() for p in pickerList]):
    Done = False
    while not Done:
        plt.pause(0.1)
        Done = plt.waitforbuttonpress()

    lmCoords = [p.cords for p in pickerList]
    lengths = [len(l) for l in lmCoords]
    if min(lengths) != max(lengths):
        raise Exception('Lists of landmarks were not consistent for each image, start over!')

    for p in pickerList:
        plt.close(p.fig)

    landmarks = np.array(lmCoords).swapaxes(0, 1).tolist()

    if len(pickerList) == 1:
        return landmarks

    for lm in landmarks:
        lm[0] = lm[0][::-1]
        lm[1] = lm[1][::-1]

    return landmarks
