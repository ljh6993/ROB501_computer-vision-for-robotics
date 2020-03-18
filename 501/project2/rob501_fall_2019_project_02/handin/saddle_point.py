import numpy as np
from numpy.linalg import inv, lstsq


def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then
    finding the critical point of that paraboloid.

    Note that the location of “'p' is relative to (-0.5, -0.5) ”at the upper
    left corner of the patch, i.e., the pixels are treated as covering an
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    # --- FILL ME IN ---

    m, n = I.shape

    # B = I.reshape((m*n,))
    # print(B)
    B=[]
    # x2 xy y2 x y 1 (m,n) begin from 0,0
    A = np.empty((0, 6))
    for x in range(0,n):
        for y in range(0,m):
            A = np.vstack((A, [x ** 2, x * y, y ** 2, x, y, 1.]))
            B.append(I[y, x])
    # w=(a,b,c,d,e,f) for parameters
    w = lstsq(A, B,rcond=None)[0]

    (a, b, c, d, e, f) = w
    # solve the intersection
    # [2a b],[b 2c]
    C = np.array([[float(2 * a), float(b)], [float(b), float(2 * c)]])

    D = np.array([-d, -e])

    pt = inv(C).dot(D)
    pt=pt.reshape(2,1)
    print(pt)

    # ------------------

    return pt



