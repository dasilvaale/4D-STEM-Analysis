# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:13:37 2020
 
@author: Alessandra da Silva: dasilvaalessandra@gmail.com

All use of this code must cite the following works:
    [1] The chain of chirality transfer in tellurium nanocrystals, Ben-Moshe, da Silva et al., under review.
    
"""

from ncempy.io import dm
from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interactive
from IPython.display import display


def parse_dm4(fname):
    """
    Parse dm4 data into 4D ndarray.

    Parameters
    ----------
    fname: str
        Path to .dm4 file.

    Returns
    -------
    dm: dict
        Dictionary of data and metadata etc.
        Data is (j, i, x, y) ndarray of 4D data of dataset 0 in the file.
        (j, i) are probe positions and (x, y) are the coordiantes of the image data.
        Data accessed as dm['data']

    """

    # Reads dm4 data and reshape into 4D stack

    dm1 = dm.fileDM(fname)
    dm1.parseHeader()
    im1 = dm1.getDataset(0)

    # print(dm1.allTags)

    scanI = int(
        dm1.allTags[".ImageList.2.ImageTags.Series.nimagesx"]
    )  # number of images x
    scanJ = int(
        dm1.allTags[".ImageList.2.ImageTags.Series.nimagesy"]
    )  # number of images y
    numkI = im1["data"].shape[2]  # number of pixels in x
    numkJ = im1["data"].shape[1]  # number of pixels in y

    im1["data"] = im1["data"].reshape([scanJ, scanI, numkJ, numkI])
    im1["metadata"] = dm1.allTags

    return im1


def plot_dm4(data, image=None, circles=None, points=None, pixel_size=None):

    """
    Plot dm4 images
    Parameters
    ----------
    data: dm4 file.
    circles: Laue circles from fit
    Points: Detected peaks for the Laue Circle
    Pixel_size from the data

    Returns
    -------
    Slider with dataset images
    """

    # interactive figure w/ circle
    fig = plt.figure(figsize=(10, 4))

    if image is not None:
        ncols = 2
    else:
        ncols = 1

    # define grid based on whether image input or not and set up axes
    gs = GridSpec(1, ncols)
    ax1 = fig.add_subplot(gs[0, 0])

    if image is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(image)
        (p2,) = ax2.plot([], [], marker="X", ls="None", ms=20, mew=3, mec="k", mfc="w")
        ax2.set_xticks([])
        ax2.set_yticks([])

    # plot 1st diffraction pattern
    im1 = ax1.imshow(data[0, 0, ...], cmap="inferno")

    if circles is not None:
        circle = plt.Circle((0, 0), 0, color="r", fill=False)
        ax1.add_artist(circle)

    if points is not None:
        (p1,) = ax1.plot([], [], "r.")

    ax1.set_xticks([])
    ax1.set_yticks([])

    def axUpdate(i, j):
        # interactive function that updates plots when sliders are changed
        d = data[j, i, ...]
        im1.set_array(d)
        im1.set_clim(d.min(), d.max())

        if image is not None:
            p2.set_xdata(i)
            p2.set_ydata(j)

        if circles is not None:
            circle.set_radius(circles[j, i, 0])
            circle.center = circles[j, i, 1:][::-1]

        if points is not None:
            p1.set_xdata(points[j, i, :, 1])
            p1.set_ydata(points[j, i, :, 0])

    display(
        interactive(
            axUpdate,
            i=IntSlider(data.shape[1] // 2, 0, data.shape[1] - 1),
            j=IntSlider(data.shape[0] // 2, 0, data.shape[0] - 1),
        )
    )

    fig.tight_layout()
    fig.show()


def plot_dm4_lattice(data, points=None, lattices=None):
    """

    Plot dm4 images.

    Parameters
    ----------
    data: dm4 file.
    lattices: Fitted lattices as list of arrays
        The first two dimensions of each array must be the same as data.

    Returns
    -------
    Slider with dataset images

    """

    # coded quick and nasty, but it works

    # interactive figure w/ circle
    fig, ax1 = plt.subplots(figsize=(10, 4))

    # plot 1st diffraction pattern
    im1 = ax1.imshow(data[0, 0, ...], cmap="inferno")

    if lattices is not None:
        lines = [[] for l in lattices]

        colors = ["w", "r", "b", "k"]
        for idx, l in enumerate(lattices):
            for i in range(-n, n + 1):
                lines[idx].append(ax1.plot([], [], color=colors[idx], alpha=0.6)[0])

            for j in range(-n, n + 1):
                lines[idx].append(ax1.plot([], [], color=colors[idx], alpha=0.6)[0])

    if points is not None:
        (p1,) = ax1.plot([], [], "r.")

    ax1.set_xticks([])
    ax1.set_yticks([])

    def axUpdate(i, j):
        # interactive function that updates plots when sliders are changed
        d = data[j, i, ...]
        im1.set_array(d)
        im1.set_clim(d.min(), d.max())

        if lattices is not None:
            for idx, lattice in enumerate(lattices):
                if np.isnan(lattice[j, i]).any():
                    for line in lines[idx]:
                        line.set_xdata([])
                        line.set_ydata([])
                else:
                    p0 = lattice[j, i].ravel()
                    for ii in range(-n, n + 1):
                        start = p0[-2:] + ii * p0[:2]
                        pts = start + (p0[2:4] * -n, p0[2:4] * n + 1)
                        lines[idx][ii + n].set_xdata(pts[:, 0])
                        lines[idx][ii + n].set_ydata(pts[:, 1])

                    for jj in range(-n, n + 1):
                        start = p0[-2:] + jj * p0[2:4]
                        pts = start + (p0[:2] * -n, p0[:2] * n + 1)
                        lines[idx][jj + 3 * n].set_xdata(pts[:, 0])
                        lines[idx][jj + 3 * n].set_ydata(pts[:, 1])

        if points is not None:
            p1.set_xdata(points[j, i, :, 1])
            p1.set_ydata(points[j, i, :, 0])

    display(
        interactive(
            axUpdate,
            i=IntSlider(data.shape[1] // 2, 0, data.shape[1] - 1),
            j=IntSlider(data.shape[0] // 2, 0, data.shape[0] - 1),
        )
    )

    fig.tight_layout()
    fig.show()
