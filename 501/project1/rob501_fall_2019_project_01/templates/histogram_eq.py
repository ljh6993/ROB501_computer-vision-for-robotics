import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.
    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---
    # method two
    # calculate histogram
    hists = np.histogram(I)

    # caculate cdf(cumulative distribution function)
    input_image = I.flatten()
    hists, binedge = np.histogram(I, bins=256, range=(0, 256))
    # hists.shape
    hists_cumsum = np.cumsum(hists)
    totalpixel = np.sum(hists)
    # build a point operation function, f
    f = (255 * (hists_cumsum) / (totalpixel))  # make sure the array is unit8
    f = np.round(f).astype('uint8')
    # mapping
    J = f[I]

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # ------------------

    return J