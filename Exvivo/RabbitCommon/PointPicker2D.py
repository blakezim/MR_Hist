# import PyCACalebExtras.SetBackend

# plt = PyCACalebExtras.SetBackend.SetBackend('GTK3agg')
# plt = PyCACalebExtras.SetBackend.SetBackend()
# import PyCA.Core as ca
# import PyCACalebExtras.Common as cc
# import PyCACalebExtras.Display as cd
# import PyCA.Common as common
# import matplotlib.image as im

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from scipy.spatial import distance

plt.ion()
plt.close('all')


def LandmarkPicker(imList):
    """Allows the user to select landmark correspondences between any number of images.
    The images must be in a list and must be formatted as numpy arrays. """

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


def ReviseLandmarks(imlist, landmarks):
    """A function for updating already chosen landmarks in the imlist.
    landmarks should be a list in the format of (Npoints X Ndimensions X Nimages).
    imlist is a list of images in numpy format.
    The order of the images in imlist should correspond to the order of the landmarks in the Nimages dimension."""

    class PointRevise(object):
        '''Image class for revising landmarks'''

        def __init__(self, X, lm):
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.im = self.ax.imshow(X, cmap='gray')
            self.shape = np.shape(X)
            self.lm = np.array(lm)
            self.ax.scatter(self.lm[:, 0], self.lm[:, 1])
            self.ax.set_xlim([0, self.shape[1]])
            self.ax.set_ylim([self.shape[0], 0])
            self.poi = []
            self.coords = self.lm.tolist()
            plt.pause(0.001)
            self.fig.canvas.mpl_connect('key_press_event', self.spress)

        def spress(self, event):
            if event.key == 'n':
                self.poi = self.lm[distance.cdist([(event.xdata, event.ydata)], self.lm).argmin()]
                print(self.poi)
                self.coords = self.coords.remove([self.poi[0], self.poi[1]])
                self.ax.remove()
                self.ax.scatter(self.poi[0], self.poi[1], c=(1, 0, 0, 1))
                self.ax.scatter(np.array(self.coords)[:, 0], np.array(self.coords)[:, 1])
                self.coords = self.lm.tolist()

    plt.ion()
    landmarks = np.array(landmarks).swapaxes(0, 1).tolist()

    reviseList = [PointRevise(im, lm) for im, lm in zip(imlist, landmarks)]

    while all([r.fig.canvas.get_visible() for r in reviseList]):
        plt.pause(0.1)

    for r in reviseList:
        plt.close(r.fig)

    return landmarks
