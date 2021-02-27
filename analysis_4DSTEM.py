# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:13:37 2020
 
@author: Alessandra da Silva: dasilvaalessandra@gmail.com

All use of this code must cite the following works:
    [1] The chain of chirality transfer in tellurium nanocrystals, Ben-Moshe, da Silva et al., under review.
    
"""


from matplotlib.gridspec import GridSpec
import numpy as np
from skimage import exposure, feature, transform
from scipy import optimize, ndimage
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from thresholding_functions import threshold_otsu_median

from ipywidgets import interactive, IntSlider, Checkbox, Dropdown, VBox
from IPython.display import display
from logging import warning


def fit_circle_through_3_points(ABC):
    """

    Fits a geometric circle through three xy pts: ABC.

    Parameters
    ----------
    ABC: (3, 2) ndarray
        3 rows of (x, y) points.

    Returns
    -------
    R: float
        Fitted radius.
    center: (2,) ndarray
        (xc, yc) fitted circle center.

    Notes
    -----
    Adapted from some MATLAB code.

    """

    x1 = ABC[0, 0]
    x2 = ABC[1, 0]
    x3 = ABC[2, 0]

    y1 = ABC[0, 1]
    y2 = ABC[1, 1]
    y3 = ABC[2, 1]

    xcyc = np.zeros([2, 1])

    mr = (y2 - y1) / (x2 - x1)
    mt = (y3 - y2) / (x3 - x2)

    #     % ============= Compute xc, the circle center x-coordinate

    xcyc[0, 0] = (mr * mt * (y3 - y1) + mr * (x2 + x3) - mt * (x1 + x2)) / (
        2 * (mr - mt)
    )

    #     % ============= Compute yc, the circle center y-coordinate
    xcyc[1, 0] = -1 / mr * (xcyc[0, 0] - (x1 + x2) / 2) + (y1 + y2) / 2

    #     % ============= Compute the circle radius
    R = np.sqrt((xcyc[0, 0] - x1) ** 2 + (xcyc[1, 0] - y1) ** 2)
    #         R(idf34) = Inf; % Failure mode (3) or (4) ==> assume circle radius infinite for this case

    return (R, xcyc)


def circle_residuals(params, points, weight=None):
    """

    Calculates sum of residuals between experimental points and circle.

    Residuals calulated as the square difference between points and radius.

    Parameters
    ----------
    params: array-like
        Circle parameters of the form (radius, *center).
        eg. in 2D (r, i, j).
    points: (N, M) ndarray
        Columns of coordinates. eg. (i, j) in 2D.
    weight: None or (N,) array-like
        Weighting for each point in points.
        If None then weighting is uniform.

    Returns
    -------
    residuals: float
        Sum of squares of residuals.

    """
    # check inputs
    points = np.asarray(points)
    if weight is None:
        weight = np.ones(len(points))
    else:
        weight = np.asarray(weight)

    # residuals are difference between (pts-center) and radius
    # weighting applied as reciprocal, ie. higher weight for smaller residual value
    residuals = (np.linalg.norm(points - params[1:], axis=1) - params[0]) * weight
    return np.square(residuals).sum()


def constrain_perimeter(params, point):
    """

    Calculates distance between circle center defined by params and a point center.
    Useful to constrain minimization functions.
    Example: Distance from fit circle to the direct beam is equal to the radius.

    Parameters
    ----------
    params: array-like
        Circle parameters of the form (radius, *center).
        eg. in 2D (r, i, j).
    point: array-like
        Center coordinates. eg. (i, j) in 2D.

    Returns
    -------
    distance: float
        Distance between poicircle defined by params and center.

    """
    # format inputs
    params = np.asarray(params)
    point = np.asarray(point)
    # return vector normal of difference
    return np.linalg.norm(params[1:] - point) - params[0]


def fit_laue_circles_3points(data, points, mask, cm0):
    """
    ALGEBRAIC SOLUTION
    Fits Laue circle to 4D-STEM diffraction data using a geometric 3-point fit method.
    Data is composed of a 2D diffraction pattern at each probe position (j, i).

    Parameters
    ----------
    data: (j, i, x, y) ndarray
        4D STEM data, last 2 dimensions are diffraction pattern dimensions.
    points: (j, i, N, 2) ndarray
        coordinates of the detected N diffraction disks for each probe position.
    mask: (j, i) ndarray of bool
        Mask of region of interest. True where points are valid.
    cm0: (2,) array-like
        Beam direction coordinates on diffraction patterns.

    Returns
    -------
    out: (j, i, 3) ndarray
        Circle parameters for Laue circle fit using geometric 3-point method.
        Last dimension is composed of (r, xc, yc).
        (j, i) pixels not fitted will have np.nan values.

    """
    # out shape [j,i,3]
    out = np.zeros(data.shape[:2] + (3,), dtype=np.float)
    out.fill(np.nan)

    for j in range(data.shape[0]):
        for i in range(data.shape[1]):

            if not mask[j, i]:  # if background ignore and pass the loop
                continue

            # get valid (non NaN) points only
            p = points[j, i][np.logical_not(np.isnan(points[j, i]).any(axis=1))]

            # get central peak, axis = 1 is the direction where linalg is been calculated
            center = p[np.argmin(np.linalg.norm(p - cm0, axis=1))]

            # distance of the peaks to the central peak
            dists = np.linalg.norm(p - center, axis=1)

            # get position of the most distant peak from center
            max1 = p[np.argmax(dists)]

            # dist_to_furthest_peak from max1
            max2 = p[np.argmax(np.linalg.norm(p - max1, axis=1))]

            # do three point circle fit
            # force direct beam peak in fit pts
            r, (x0, y0) = fit_circle_through_3_points(np.array([max1, center, max2]))

            # store result in array
            out[j, i, :] = (r, x0, y0)

    return out


def detect_disks(data, mask=None, num_peaks=20, logscale=True, verbose=True, **kwargs):
    """

    Detect diffraction disks on 4D-STEM data for each probe position (j, i).

    Parameters
    ----------
    data: (j, i, x, y) ndarray
        4D-STEM data where first two dimensions are probe position (j, i) and latter two are dimensions of each diffraction image.
    mask: (j, i) ndarray of bool
        Mask of region of interest. True where points are valid.
    num_peaks: int
        Maximum number of peaks to return.
    logscale: bool
        If True the image is rescaled and peaks are logarithm transformed to increase image contrast.

    kwargs_plm:
        Passed to skimage.feature.blob_log

    Returns
    -------
    peaks: (j, i, N, 2) ndarray
        Diffraction disk locations for each probe position.

    """

    # define default values for kwargs
    kwargs.setdefault("min_sigma", 5)
    kwargs.setdefault("max_sigma", 10)
    kwargs.setdefault("num_sigma", 6)
    kwargs.setdefault("threshold", 0.1)

    if mask is not None:
        assert (
            mask.shape == data.shape[:2]
        ), "mask must have same shape as first two dimensions of data."

    # initialize array with NaN values
    ndim = len(
        data.shape[2:]
    )  # data dimension not including scanning pixels, should be 2
    out = np.empty((data.shape[0], data.shape[1], num_peaks, ndim), dtype=np.float)
    out.fill(np.nan)  # fill array with NaN, real values will overwrite these

    for j in range(data.shape[0]):
        for i in range(data.shape[1]):

            print("Fitting: {} {}".format(j, i), end="\t\t\t\r")

            # if background, ignore and continue the loop
            if mask is not None and not mask[j, i]:
                continue

            # minimum subtraction +1 stops negative numbers from crashing log
            if logscale:
                image = exposure.rescale_intensity(
                    np.log(data[j, i, ...].astype(float) - data[j, i].min() + 1.0),
                    out_range=(0.0, 1.0),
                )
            else:
                image = exposure.rescale_intensity(
                    data[j, i, ...].astype(float), out_range=(0.0, 1.0)
                )

            # find peaks
            peaks = feature.blob_log(image, **kwargs)
            # sort peaks by most intense
            # as just a relative sort, also works on log image the same

            intensities = data[j, i][
                tuple(
                    aa.ravel()
                    for aa in np.split(peaks[:, :-1].astype(int), ndim, axis=1)
                )
            ]
            peaks = peaks[np.argsort(intensities)[::-1]]
            # allocate up to num_peaks of detected peaks
            out[j, i][: len(peaks)] = peaks[:num_peaks, :-1]

    return out


def detect_disks_hough(
    data, mask, hough=False, sigma=5.0, kwargs_plm={}, kwargs_hough={}
):
    """

    Detect diffraction disks on 4D-STEM data for each probe position (j, i).

    Parameters
    ----------
    data: (j, i, x, y) ndarray
        4D-STEM data where first two dimensions are probe position (j, i) and latter two are diffraction data.
    mask: (j, i) ndarray of bool
        Mask of region of interest. True where points are valid.
    hough: bool
        Detect peaks using Hough circles.
    sigma: float
        Sigma for skimage.feature.canny, used if hough is True.
    kwargs_plm:
        Passed to skimage.feature.peak_local_max.
    kwargs_hough:
        Passed to skimage.transform.hough_circle_peaks.

    Returns
    -------
    peaks: (j, i, N, 2) ndarray
        Diffraction disk locations for each probe position.

    """

    # define default values for kwargs
    kwargs_plm.setdfault(dict(min_distance=10, threshold_rel=0.2, num_peaks=10))

    kwargs_hough.setdefault(
        dict(radii=np.arange(5, 30), total_num_peaks=kwargs_plm["num_peaks"])
    )

    for j in range(data.shape[0]):
        for i in range(data.shape[1]):

            if mask[j, i]:  # if background ignore and pass the loop
                continue

            image = exposure.rescale_intensity(
                np.log(data[j, i, ...] - data[j, i].min() + 1).astype(float),
                out_range=(0.0, 1.0),
            )

            # find peaks
            peaks = feature.peak_local_max(image, indices=True, **kwargs_plm)

            # refine peaks using hough circle transform
            if hough:
                acc, cx, cy, radii = transform.hough_circle_peaks(
                    transform.hough_circle(
                        feature.canny(image, sigma=sigma), kwargs_hough["radii"]
                    ),
                    **kwargs_hough,
                )

                peaks = np.column_stack((cy, cx))

    return peaks


def fit_laue_circles(
    data, mask, cm0, guess, points, constrain=True, verbose=True, **kwargs
):
    """

    Fits Laue circle to 4D-STEM diffraction data using minimization.
    Data is composed of a 2D diffraction pattern at each probe position (j, i).

    Parameters
    ----------
    data: (j, i, x, y) ndarray
        4D STEM data, latter 2 dimensions are diffraction pattern dimensions.
    mask: (j, i) ndarray of bool
        Mask of region of interest. True where points are valid.
    guess: (j, i, 3) ndarray
        Initial guess of circle for each probe position. Last dimension is (xc, yc, r).
    points: (j, i, N, 2) ndarray
        Locations of N diffraction disks for each probe position.
    constrain: bool, default is True
        If True fitted circle perimeter is constrained to pass through cm0.
    verbose: bool
        Verbose output printed to console if True.
    kwargs:
        Passed to scipy.optimize.minimize.

    Returns
    -------
    out: (j, i, 3) ndarray
        Minimized Laue circle fit parameters.
        (j, i) positions not fitted will have np.nan values.

    """
    # set dfault values for minimization
    kwargs.setdefault("method", "SLSQP")

    if constrain:
        kwargs.setdefault(
            "constraints", {"fun": constrain_perimeter, "type": "eq", "args": (cm0,)}
        )

    # to hold circle params obtained from circle fit through minimization
    out = np.zeros_like(guess, dtype=np.float)
    out.fill(np.nan)

    # for each probe position
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):

            if verbose:
                print("Fitting {} {}".format(j, i), end="\t\t\r")

            if not mask[j, i]:  # if background
                continue

            # get image
            image = data[j, i, ...]

            # get valid points, ie. not NaN
            p = points[j, i][np.logical_not(np.isnan(points[j, i]).any(axis=1))]

            # weighted by intensity on image
            weight = image[
                tuple(np.split(np.around(p).astype(np.int), image.ndim, axis=1))
            ]

            result = optimize.minimize(
                circle_residuals, guess[j, i, ...], args=(p, weight), **kwargs
            )

            out[j, i, :] = result.x

    return out


def get_new_ij(i, j, delta=1):
    """
    Get random neighboring ij value.

    Parameters
    ----------
    i, j: int

    Returns
    -------
    i, j: int
        Random neighbouring value.

    """

    inew = i
    jnew = j
    # perform as loop such that at least one of returned values is different
    while inew == i and jnew == j:
        inew = i + np.random.randint(-delta, delta + 1)
        jnew = j + np.random.randint(-delta, delta + 1)
    return inew, jnew


def attempt_manual_fit(cp, pts, to_fit, mask, **kwargs):
    """
    Try to do manual circle fitting. Fn chooses random neighboring pt to use as guess.

    Parameters
    ----------
    cp: numpy.ndarray
        Array of circle coords, shape = (j, i, 3) ie. (j, i, (xc, yc, r))
    pts: numpy.ndarray
        Array of found spots, shape = (j, i, n, 2)
    to_fit: array-like
        list or array of ji coords to refit
    mask: numpy.ndarray, dtype=bool
        True where pts are valid.
    **kwargs: dict
        kwargs to pass to scipy.optimize.minimize.

    Returns
    -------
    new_fmin: nump.ndarray
        New fit circle coords, same shape as fmin.

    """
    # do manual fit circle parameters with near pts
    cp_new = cp.copy()

    for j, i in to_fit:
        # skip points that are not in fitting region
        if not mask[j, i]:
            continue
        jnew, inew = j, i

        # make sure we are getting a valid point (in mask)
        # and that it is not the same as point being refit (ji)
        while not mask[jnew, inew] or (jnew == j and inew == i):
            inew, jnew = get_new_ij(i, j)

            if jnew >= mask.shape[0]:
                jnew = mask.shape[0] - 1
            if inew >= mask.shape[1]:
                inew = mask.shape[1] - 1
            if jnew < 0:
                jnew = 0
            if inew < 0:
                inew = 0

        res = optimize.minimize(
            circle_residuals,
            cp[jnew, inew],
            args=(np.unique(pts[j, i, ...], axis=0)),  # fit only unique points
            **kwargs,
        )
        # add params into new array
        cp_new[j, i] = res.x

    return cp_new


def decompose_phi_from_circle(
    circle_coords, center, pixel_size, wavelength, degrees=True
):
    """
    Compute phix, phiy from circle fit coords (r,j,i).

    Parameters
    ----------
    circle_coords: circle coordinates. Standard coordinates are the output from the fit_laue_circle functions
    np.ndarray, shape = (j,i,3)
        (radius, row, col) for each ji
    center: array_like
        Bright field spot position. (row, col)
    pixel_size: float
        same units as lambda.
    wavelength:
        electron wavelength, same units as pixel_size.
    degrees: bool
        if True phix, y are returned in degrees rather than radians.

    Returns
    -------
    phix, phiy: (j, i) np.ndarray

    """

    # horizontal component, ie. j (column) in diffraction pattern
    dx = np.nan_to_num(circle_coords[..., 2] - center[1])
    # vertical component, ie. i (row) in diffraction pattern
    dy = np.nan_to_num(circle_coords[..., 1] - center[0])

    phix = dx * wavelength * pixel_size
    phiy = dy * wavelength * pixel_size

    if degrees:
        phix = np.rad2deg(phix)
        phiy = np.rad2deg(phiy)

    return phix, phiy


def rough_virtual_rec(data, box_size=150):

    """
    Perform a rough virtual bright field reconstruction from a rectangular box centered on the direct beam spot.

    Parameters
    ----------
    data: 4D (M, N, L, P) ndarray
    box_size: size of the box where the reconstruction is performed.

    Returns
    -------
    VR0: (M, N) ndarray
        Virtual reconstruction.
    cm0: center of mass of first image from the dataset
    mask: mask that is one inside nanoparticle and 0 outside

    """

    # find the center of mass given by sum(coordinate * pixel intensity)/  sum(intensities)
    cm0 = [int(i) for i in ndimage.measurements.center_of_mass(data[0, 0, ...])]

    VR0 = np.sum(
        data[
            ...,
            cm0[0] - box_size : cm0[0] + box_size,
            cm0[1] - box_size : cm0[1] + box_size,
        ],
        axis=(-2, -1),
    )
    # sums all CBED images (15x83) over the y axis, then sum all the CBED images over the x axis resulting in a single (15x83)
    # bright field image

    mask = VR0 > threshold_otsu_median(VR0)
    return VR0, cm0, mask


def ij_box_indices(points, lengths, shape):
    """
    Determines the limits of a rectangular box that surrounds a selected point. The selected point will be
    in the middle of the box.
    Parameters
    ----------
    points: a tuple with the coordinates i and j of the point that the rectangle is drawn
    lengths: a tuple with the lenghts of the rectangle at each direction
    shape: the shape of the image where the points are contained

    Returns
    --------------
    A list with the min and max rectangle coordinates

    """

    assert (
        len(points) == len(lengths) == len(shape)
    ), "points, lengths, and shape must be defined for each dimension."

    out = []

    for k in range(len(points)):
        _min = points[k] - lengths[k] // 2
        _max = points[k] + lengths[k] // 2

        if _min < 0:
            _min = 0
        if _max > shape[k] - 1:
            _max = shape[k] - 1

        out.append((_min, _max))

    return out


def generate_VR(data, i, j, i_side, j_side, in_box=True):

    """
    Generate virtual reconstruction from set of difraction disks using a rectangular virtual aperture
    Parameters
    ----------
    data: (i, j, x, y) ndarray
        4D STEM data, latter 2 dimensions are diffraction pattern dimensions.
    i: row coordinate of the selected diffraction disk in i
    j: column coordinate of the selected diffraction disk in j
    i_side: height of the box in the vertical direction
    j_side: width of the box in the horizontal direction

    Returns
    ------------
    Virtual Reconstructed image with dimensions (x,y)

    """

    i_range, j_range = ij_box_indices((i, j), (i_side, j_side), data.shape[-2:])

    # generate VR within box
    image = np.sum(data[..., slice(*i_range), slice(*j_range)], axis=(-2, -1))

    # otherwise return VR from whole image - VR in box
    # ie. VR outside box
    if not in_box:
        image = np.sum(data, axis=(-2, -1)) - image

    return image


def interactive_VR(data, image=None, box=50, cmap="inferno"):
    """

    Generate interactive jupyter notebook figure with adjustable slider to generate virtual reconstructions.

    Parameters
    ----------
    data, ndarry, rank-4
        4D-STEM data.

    """
    # generate figure
    fig, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1.0 / 2))

    # initial plot
    if image is None:
        ax[0].matshow(data[data.shape[0] // 2, data.shape[1] // 2, ...], cmap=cmap)
    else:
        ax[0].matshow(image, cmap=cmap)

    # rectangle overlay
    # box = 50
    rect = plt.Rectangle(
        np.array(data.shape[-2:][::-1]) - box / 2, box, box, color="w", fill=False
    )
    ax[0].add_patch(rect)

    #  initial recon dummy data -> has same shape as recon
    recon = ax[1].matshow(np.zeros(data.shape[:2]), cmap=cmap)

    # create a layer with alpha channel
    layer_shape = ax[0].images[0].get_array().shape + (4,)
    layer = ax[0].imshow(np.zeros(layer_shape))
    alpha = 0.5

    ax[0].legend(handles=[Patch(color="w", alpha=alpha, label="Reconstruction area")])

    # interactive fn
    def update(i, j, side, in_box, evaluate):
        """Do actual axes updates"""
        rect.set_xy((j - side / 2, i - side / 2))
        rect.set_width(side)
        rect.set_height(side)

        i_range, j_range = ij_box_indices((i, j), (side, side), data.shape[-2:])

        if evaluate:
            print("Generating reconstruction...", end="\t\t\r")
            image = generate_VR(data, i, j, side, side, in_box=in_box)

            print("Reconstruction generated.", end="\t\t\r")
            recon.set_array(image)
            recon.set_clim(image.min(), image.max())

        if in_box:
            temp = np.zeros(layer_shape)
            # set as white
            temp[slice(*i_range), slice(*j_range), ...] = 1
            # set opacity
            temp[slice(*i_range), slice(*j_range), -1] = alpha
        else:
            # define white layer
            temp = np.ones(layer_shape)
            # set opacity
            temp[..., -1] = alpha
            # cut out box
            temp[slice(*i_range), slice(*j_range), :] = 0

        layer.set_array(temp)

    display(
        interactive(
            update,
            i=IntSlider(data.shape[-2] // 2, 0, data.shape[-2] - 1),
            j=IntSlider(data.shape[-1] // 2, 0, data.shape[-1] - 1),
            side=IntSlider(box, 0, data.shape[-1] // 2),
            in_box=Checkbox(True),
            evaluate=Checkbox(False),
        )
    )


def interactive_fit(
    data, points, circles, cm0, constrain=True, origin="lower", **kwargs
):
    """

    Interactive Laue circle fitting function.
    Given position (j, i) will be fitted with initial guess determined by mouse click on plot.
    The array circles will be updated according to the new fits.

    Parameters
    ----------
    data: (j, i, x, y) ndarray
        4D-STEM data.
    points: (j, i, N, 2) ndarray
        Array of N points for each scan pixel (j, i).
    circles: (j, i, N, 3) ndarray
        Circle parameters from fit (r, x0, y0)
    cm0: array-like length 2
        Direct beam location.
    constrain: bool
        If True then fitted circle perimeter is constrained to pass through cm0.
    origin: str
        Origin for plot, either 'lower' or 'upper'.
    kwargs: passed to scipy.optimize.minimize

    """

    warning("Using this function will overwrite active fitting data.")

    # make cm0 array
    cm0 = np.asarray(cm0, dtype=np.float)

    # set default args for minimize
    kwargs.setdefault("method", "SLSQP")
    if constrain:
        kwargs.setdefault(
            "constraints", {"fun": constrain_perimeter, "type": "eq", "args": (cm0,)}
        )

    # create figure and initial plot
    fig, ax = plt.subplots()
    axim = ax.matshow(data[0, 0, ...], cmap="inferno", origin=origin)

    # create circle
    circle = plt.Circle((0, 0), 0, color="w", fill=False)
    ax.add_artist(circle)

    # plot points
    (pts,) = ax.plot([], [], "r.")

    def update(i, j, box):
        # interactive function
        axim.set_array(data[j, i, ...])
        axim.set_clim(data[j, i, ...].min(), data[j, i, ...].max())

        circle.center = circles[j, i, 1:][::-1]
        circle.set_radius(circles[j, i, 0])

        p = points[j, i][np.logical_not(np.isnan(points[j, i]).any(axis=1))]
        pts.set_xdata(p[:, 1])
        pts.set_ydata(p[:, 0])

    # create sliders, checkbox, and dropdown
    i = IntSlider(0, 0, data.shape[1] - 1)
    j = IntSlider(0, 0, data.shape[0] - 1)
    box = Checkbox(False, description="Evaluate")
    selection = Dropdown(options=["Fit", "Manual Circle"], description="Method")

    flag = False
    clicked_location = (0, 0)

    def on_click(event):
        """Update flag"""
        nonlocal flag, clicked_location
        flag = True
        clicked_location = np.array((event.xdata, event.ydata), dtype=np.float)

    def on_release(event):
        """Reset flag value and fit if selected"""
        nonlocal flag

        flag = False

        if box.value:
            # only do fit if Checkbox is ticked
            # make sure that 'evaluate' is True as precaution

            if selection.value == "Fit":
                # (radius, row, column)
                guess = (
                    np.linalg.norm(np.array((event.ydata, event.xdata)) - cm0),
                    event.ydata,
                    event.xdata,
                )
            elif selection.value == "Manual Circle":
                guess = (circle.get_radius(), *circle.get_center()[::-1])

            res = optimize.minimize(
                circle_residuals, guess, args=(points[j.value, i.value],), **kwargs
            )

            print(f"New fit values: {res.x}", end="\t\t\t\r")

            circles[j.value, i.value] = res.x

            # update plot
            update(i.value, j.value, box.value)

        circle.set_ls("solid")

    def on_move(event):
        """Draws manual circle if selected"""

        if flag:
            if constrain:
                pt0 = cm0
            else:
                pt0 = clicked_location

            # mouse location - cm0
            delta = -1 * (pt0 - (event.ydata, event.xdata))
            # center is halfway towards delta
            circle.set_center((pt0 + delta / 2)[::-1])
            # radius is half length of delta
            circle.set_radius(np.linalg.norm(delta) / 2)
            circle.set_ls("dashed")

    # connect callbacks
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    display(VBox([interactive(update, i=i, j=j, box=box), selection]))


def plot_laue_circle_results(data, phi1, phi2, VBF, mask, circles, points):
    """

    Interactive func to show the data and resulting fits.

    Parameters
    ----------
    data: 4d-ndarray
        The 4D-STEM data.
    phi1, phi2: 2d ndarray
        The produced maps.
    VBF: 2d ndarray
        Virtual Reconstruction.
    mask: 2d ndarray
        Mask of relevant data points.
    circles: 3d ndarray
        The Laue circle parameters for each diffraction patterns.
    points: 3d ndarray
        The computed diffraction spot positions for each diffraction pattern.

    """
    fig = plt.figure(figsize=(10, 4))

    gs = GridSpec(3, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    axphix = fig.add_subplot(gs[1, 0])
    axphiy = fig.add_subplot(gs[2, 0])

    layer = np.ones(data.shape[:2] + (4,))
    layer[mask, -1] = 0

    fraction = 0.1
    im12 = axphix.matshow(phi1, cmap="Spectral")
    axphix.imshow(layer)
    cb12 = fig.colorbar(im12, ax=axphix, fraction=fraction)

    im13 = axphiy.matshow(phi2, cmap="Spectral")
    axphiy.imshow(layer)
    cb13 = fig.colorbar(im13, ax=axphiy, fraction=fraction)

    axphix.set_axis_off()
    axphiy.set_axis_off()

    (px,) = axphix.plot([], [], "o", markersize=5, mew=2, mec="k", mfc="w")
    (py,) = axphiy.plot([], [], "o", markersize=5, mew=2, mec="k", mfc="w")

    im11 = ax1.imshow(VBF, cmap="inferno")
    cb11 = fig.colorbar(im11, ax=ax1, fraction=fraction)
    ax1.set_xticks([])
    ax1.set_yticks([])

    im2 = ax2.imshow(data[0, 0, :, :], origin="lower", cmap="gray")
    (p2,) = ax2.plot([], [], "r.")

    # # show circle solutions from minimization alogirithm
    _circle = ax2.add_patch(
        plt.Circle((0, 0), 0, lw=2, color="red", fill=None, label="Laue circle fit")
    )

    def axUpdate(i, j):

        px.set_xdata(j)
        px.set_ydata(i)
        py.set_xdata(j)
        py.set_ydata(i)

        im2.set_array(data[i, j])

        _circle.set_radius(circles[i, j, 0])
        _circle.center = (circles[i, j, 1], circles[i, j, 2])

        p2.set_data(points[i, j].T[::-1])

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.legend()

    # for all plots -->
    w = interactive(
        axUpdate,
        i=IntSlider(VBF.shape[0] // 2, 0, VBF.shape[0] - 1),
        j=IntSlider(VBF.shape[1] // 2, 0, VBF.shape[1] - 1),
    )

    display(w)
    fig.tight_layout()
    fig.show()
