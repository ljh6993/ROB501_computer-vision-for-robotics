import numpy as np

def find_jacobian(K, kappa, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera calibration.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate *and* the camera model parameters
    (intrinsics and radial distortion), given a landmark point.

    Parameters:
    -----------
    K      - 3x3 np.array, camera intrinsic calibration matrix.
    kappa  - 1x1 np.array, radial distortion coefficients (k1).
    Twc    - 4x4 np.array, homogenous pose matrix, camera pose in world frame.
    Wpt    - 3x1 np.array, world point on calibration target.

    Returns:
    --------
    JTwc - 2x6 np.array, Jacobian (columns are tx, ty, tz, r, p, q).
    JKk  - 2x5 np.array, Jacobian (columns are fx, fy, cx, cy, k1).
    """
    #--- FILL ME IN ---

    # Remember, each column in the Jacobian is the partial derivative of the
    # output of the projection function (u or v) with respect to one of the
    # model parameters.

    #------------------

    return JTwc, JKk