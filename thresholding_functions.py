# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:21:30 2020

@author: Alessandra
"""


from skimage import exposure
import numpy as np

def threshold_otsu_median(image, nbins=256):
    """
    
    Calculates a greyscale image threshold using Otsu's method but using the class median (histogram mode) as the intraclass descriptor.
    This algorithm works especially well for the edges of the image bright features.

    This '#1' version uses the class modes of the image histogram as the threshold as opposed to the class medians of the image in the '#2' version.
    As a result it is over 10x faster in testing.
    The thresholds obtained from '#1' and '#2' are extremely similar.

    Parameters
    ----------
    image: np.ndarray
        The image to threshold.
    nbins: int
        Number of bins for the histogram. Default is 256.

    Returns
    -------
    threshold: float
        The calculated threshold.
    """
    # image needed as float, otherwise median calculations stall
    image = image.astype(float)
    hist, bin_centers = exposure.histogram(image, nbins=nbins)

    # class probabilities for all possible thresholds
    # NB. float-type conversion needed for python2 support, as hist.dtype == int
    weight1 = np.cumsum(hist) / np.sum(hist).astype(float)
    weight2 = np.cumsum(hist[::-1])[::-1] / np.sum(hist).astype(
        float
    )  # cumulative sum backwards

    # class medians for all possible thresholds
    mode1 = bin_centers[np.array([np.argmax(hist[:i]) for i in range(1, hist.size)])]
    mode2 = bin_centers[
        np.array([i + np.argmax(hist[i:]) for i in range(1, hist.size)])
    ]

    # median for whole image
    modeT = bin_centers[np.argmax(hist)]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = (
        weight1[:-1] * (mode1 - modeT) ** 2 + weight2[1:] * (mode2 - modeT) ** 2
    )

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold