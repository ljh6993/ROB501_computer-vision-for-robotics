import numpy as np

def camera_forward_3D_to_2D(K, kappa, Twc, Wpt):
    """
    World (target) point in 3D to image plane point in 2D.

    The function computes the projection of the 3D point 'Wpt' onto the camera
    image plane, for a distorted (warped) camera model.

    Parameters:
    -----------
    K      - 3x3 np.array, camera intrinsic calibration matrix.
    kappa  - 1x1 np.array, radial distortion coefficient (k1).
    Twc    - 4x4 np.array, homogenous pose matrix, camera pose in world frame.
    Wpt    - 3x1 np.array, world point on calibration target.

    Returns:
    --------
    p  - 2x1 np.array, projection on image plane (u, v) (horz., vert.).
    """
    #--- FILL ME IN ---

    # Transform world point to camera frame.

    # Compute normalized coordinates.

    # Compute distorted coordinates.

    # Compute image plane coordinates.

    #------------------

    return p