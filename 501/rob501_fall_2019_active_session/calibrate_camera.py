import glob
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from mat4py import loadmat
from solve_calibration_nls import solve_calibration_nls

def calibrate_camera(I_files, bboxes, Wpts):  # You may add addtional params...
    """
    Main calibration function/program.

    *NOTE:* If you are unable to optimize a specific camera pose, you may
    omit it - in this case, return a 4x4 np.array of zeros for that pose.

    Returns:
    --------
    K         - 3x3 np.array, refined intrinsic matrix (now calibrated!)
    kappa     - 1x1 np.array, refined radial distortion coefficient.
    Twc_list  - List of 4x4 np.arrays, refined pose matrices, camera poses
                in target frame (for each camera).
    """

    # Insert code as needed here...

    # Load each image.
    for I_file in I_files:
        I = imread(I_file)
        bbox = np.array(bboxes[I_file[8:-4]])

        # Comment out code below later on...
        plt.imshow(I, cmap = "gray")
        plt.plot(bbox[0, :], bbox[1, :], 'o', c='r',)
        plt.show()

    # Insert code as needed here...

    # Perform full calibration.
    return solve_calibration_nls(Kg, kappag, Twcg_list, Ipts_list, Wpts)

if __name__ == "__main__":
    # Grab the names of all image files.
    images = sorted(glob.glob("targets/*.png"))
    print(images)  

    # Load the bounding boxes.
    bboxes = loadmat("bboxes.mat")  # You will need to convert to np.arrays
    print(bboxes)

    # Load the world points.
    Wpts = np.array(loadmat("world_pts.mat")["world_pts"])
    print(Wpts)

    # Run calibration.
    results = calibrate_camera(images, bboxes, Wpts)
    print(results)