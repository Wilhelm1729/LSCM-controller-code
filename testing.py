import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import random

import translation_finder


def find_translation(img1, img2):
    """
    Given two 2D NumPy arrays, identifies the translational difference between the two and returns
    the shift in x- and- y-coordinates that is needed to reposition img2 to coincide with img1.

    Inputs:
    img1 (2D NumPy Array): The "reference" image, which the other image should be moved around to maximally coincide with.
    img2 (2D Numpy Array): The "shifted" image, which can be repositioned using the output coordinate shifts to align with the reference image.

    Outputs:
    x_shift (int): The amount of pixels that img2 should be shifted to the right in order to maximally coincide with img1.
    y_shift (int): The amount of pixels that img2 should be shifted up in order to maximally coincide with img1.
    """

    # Create correlation map
    corr = signal.correlate2d(img1, img2, mode="same")

    # The center coordinates of the shifted imgage
    y2, x2 = np.array(img2.shape) // 2

    # The coordinates in the reference images that maximally correlate with center coordinates of shifted image
    y, x = np.unravel_index(np.argmax(corr), corr.shape)

    x_shift = x2 - x
    y_shift = y2 - y

    # Visualize results
    fig, (ax_img1, ax_img2, ax_corr) = plt.subplots(1, 3, figsize=(15, 5))
    im = ax_img1.imshow(img1, cmap="gray")
    ax_img1.set_title("img1")
    ax_img2.imshow(img2, cmap="gray")
    ax_img2.set_title("img2")
    im = ax_corr.imshow(corr, cmap="viridis")
    ax_corr.set_title("Cross-correlation")
    ax_img1.plot(x, y, "ro")
    ax_img2.plot(x2, y2, "go")
    ax_corr.plot(x, y, "ro")
    fig.show()

    plt.waitforbuttonpress()

    return x_shift, y_shift


img1 = np.load("kex\LSCM-controller-code\Matrix_3.npy")
img2 = np.load("kex\LSCM-controller-code\Matrix_3_shifted.npy")

x_shift, y_shift = find_translation(img1, img2)

print(f"x_shift = {x_shift}, y_shift = {y_shift}")
