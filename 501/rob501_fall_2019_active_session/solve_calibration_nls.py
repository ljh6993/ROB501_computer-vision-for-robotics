import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def solve_calibration_nls(Kg, kappag, Twcg_list, Ipts_list, Wpts):
    """
    Estimate camera poses *and* calibration parameters from 2D-3D 
    correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera poses in the calibration
    target frame and the camera calibration parameters, given 2D-3D point 
    correspondences.

    *NOTE:* If you are unable to optimize a specific camera pose, you may
    omit it - in this case, return a 4x4 np.array of zeros for that pose.

    Parameters:
    -----------
    Kg         - 3x3 np.array, initial guess for camera intrinsic matrix.
    kappag     - 1x1 np.array, guess for radial distortion coefficient (k1).
    Twcg_list  - List of 4x4 np.array, pose matrices, guesses for camera poses.
    Ipts_list  - List of 2xn np.array, cross-junctions (with subpixel resolution).
    Wpts       - 3xn np.array of known world points (one-to-one with Ipts).

    Returns:
    --------
    K         - 3x3 np.array, refined intrinsic matrix (now calibrated!)
    kappa     - 1x1 np.array, refined radial distortion coefficient.
    Twc_list  - List of 4x4 np.arrays, refined pose matrices, camera poses
                in target frame (for each camera).
    """
    maxIters = 250                          # Set maximum iterations.

    #--- FILL ME IN ---

    # Initialize your parameter vector. What are the parameters? How many.
    # This should be a column vector.

    # Intialize your Jacobian matrix - you may fill it with zeros initially.
    # The Jacobian should have n rows, where n is equal to the total number
    # of obsevations (each target point in each image gives two observations).

    # Initialize your vector of residuals (observed - predicted) to zeros.
    # The residuals vectors should have n rows, too.

    iters = 1

    while True:
        pass

        # Project each target point into image, given current pose estimate.
        for i in np.arange(0):
            pass
            # Project point...

            # Compute residual and fill in residual vector at right spot...

            # Compute Jacobian submatrix (in two parts) for *this* obseveation
            # and insert the pieces in the proper places in the full Jacobian.

        # Solve system of normal equations for this iteration...
        # Then update parameter estimates.

        # Check - has the estimate converged?
        if converged:
            print("Covergence required %d iters." % iters)
            break
        elif iter == maxIters:
            print("Failed to converge after %d iters." % iters)
            break

        iters += 1

    #------------------
    
    return K, kappa, Twc_list 

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T