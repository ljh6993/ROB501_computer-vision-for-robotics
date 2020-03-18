import numpy as np
from numpy.linalg import inv


def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer
    #(use round()).

    This function is for a *single* image band only - for RGB images, you will
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    # --- FILL ME IN ---
    y = pt[0, 0]
    x = pt[1, 0]

    # for y, y1 y2
    rounddown = int(round(y - 0.5))
    roundup = int(round(y + 0.5))
    # for x， x1, x2
    roundleft = int(round(x - 0.5))
    roundright = int(round(x + 0.5))
    print(roundup, rounddown, roundleft, roundright)

    pixel_11 = I[roundleft, rounddown]
    pixel_12 = I[roundleft, roundup]
    pixel_22 = I[roundright, roundup]
    pixel_21 = I[roundright, rounddown]

    # R1 = ((x2 – x)/(x2 – x1))*Q11 + ((x – x1)/(x2 – x1))*Q21
    R1 = ((roundright - x) / (roundright - roundleft)) * pixel_11 + (
                (x - roundleft) / (roundright - roundleft)) * pixel_21
    # R2 = ((x2 – x)/(x2 – x1))*Q12 + ((x – x1)/(x2 – x1))*Q22
    R2 = ((roundright - x) / (roundright - roundleft)) * pixel_12 + (
                (x - roundleft) / (roundright - roundleft)) * pixel_22

    # b = ((y2 – y)/(y2 – y1))*R1 + ((y – y1)/(y2 – y1))*R2
    b = ((roundup - y) / (roundup - rounddown)) * R1 + ((y - rounddown) / (roundup - rounddown)) * R2
    b = int(round(b))

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # ------------------

    return b