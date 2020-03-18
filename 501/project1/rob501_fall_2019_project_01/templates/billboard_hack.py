# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    -----------

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    Ist_eq = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Main 'for' loop to do the warp and insertion -
    # this could be vectorized to be faster if needed!
    # bbox [x][y]
    # adjust the path a little bit. change the bbox [1,1] plus 20
    path = Path(
        [(bbox[0, 0], bbox[1, 0]), (bbox[0, 1], bbox[1, 1] + 20), (bbox[0, 3], bbox[1, 3]), (bbox[0, 2], bbox[1, 2]),
         (bbox[0, 0], bbox[1, 0])], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    # (x, y)  y is the perpendicular coordinate

    for y in range(Iyd.shape[0]):
        for x in range(Iyd.shape[1]):
            if path.contains_point([x, y]) == 1:
                # calculate the coordinate in picture Ist
                [u, v, w] = np.dot(H, [x, y, 1])
                [u, v, w] = [u, v, w] / w
                # if (u < Ist.shape[1] and v < Ist.shape[0]):
                if (u < 219 and v < 410):
                    Ihack[y, x] = bilinear_interp(Ist_eq, np.array([[u, v]]).T)

    #------------------

    # plt.imshow(Ihack)
    # plt.show()
    # imwrite(Ihack, 'billboard_hacked.png');

    return Ihack
