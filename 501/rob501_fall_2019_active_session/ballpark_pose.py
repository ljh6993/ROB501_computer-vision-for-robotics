import numpy as np

def ballpark_pose(I, bbox, Wpts):  # You may add addtional params...
    """
    Compute a 'ballpark' camera pose estimate.

    The function computes a coarse, ballpark estimate of the camera
    pose Twc. Remember that the frame attached to the target is
    oriented with x pointing along the direction with a larger number
    of squares, y pointing along the direction with fewer squares,
    and z pointing away from the camera (in general).

    Parameters:
    -----------
    I     - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bbox  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts  - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Twcg  - 4x4 np.array, pose matrix, rough guess for camera pose.
    """
    #--- FILL ME IN ---

    #------------------

    return Twcg