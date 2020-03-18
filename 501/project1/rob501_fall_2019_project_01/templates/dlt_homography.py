import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    #build a  homography matrix H
    H = np.zeros((9, 1))
    A = np.zeros((8, 9))
    for i in range(0, 4):
        # first row
        A[i * 2, 0] = -I1pts[0, i]
        A[i * 2, 1] = -I1pts[1, i]
        A[i * 2, 2] = -1
        A[i * 2, 6] = I1pts[0, i] * I2pts[0][i]
        A[i * 2, 7] = I1pts[1, i] * I2pts[0][i]
        A[i * 2, 8] = I2pts[0][i]
        # load for second row
        A[i * 2 + 1, 3] = -I1pts[0, i]
        A[i * 2 + 1, 4] = -I1pts[1, i]
        A[i * 2 + 1, 5] = -1
        A[i * 2 + 1, 6] = I1pts[0, i] * I2pts[1][i]
        A[i * 2 + 1, 7] = I1pts[1, i] * I2pts[1][i]
        A[i * 2 + 1, 8] = I2pts[1][i]

    b = np.zeros((1, 8))
    H = null_space(A)

    # normolize H
    v = H[8]
    H = H / v
    H = np.resize(H, (3, 3))
    # H = np.linalg.svd(A, b,full_matrices=True)

    return H, A


