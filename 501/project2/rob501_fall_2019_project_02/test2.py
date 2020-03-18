import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    #--- FILL ME IN ---

    m, n = I.shape
    #assume we have I that is always X-corner
    A_list = []
    B_list = []
    #iterate and create the solve for linear least square
    for x in range(n):
        for y in range(m):
            A_list.append([x**2,x*y ,y**2, x,y, 1.])
            B_list.append(I[y,x])
    A = np.array(A_list)
    B = np.array(B_list)
    #calculate least square
    a,b,c,d,e,f = lstsq(A,B,rcond=None)[0]
    print( a,b,c,d,e,f)
    #reconfigure the pts
    pt = -np.matmul(inv(np.array([[2*a, b],[b,2*c]])),np.array([d,e])).reshape((2,1))
    #------------------
    return pt

# Build non-smooth but noise-free test patch.
Il = np.hstack((np.ones((10, 7)), np.zeros((10, 13))))
Ir = np.hstack((np.zeros((10, 8)), np.ones((10, 12))))
I = np.vstack((Il, Ir))

pt = saddle_point(I)
print(pt.shape)

print('Saddle point is at: (%.2f, %.2f)' % (pt[0], pt[1]))


def boundary_delete(array, bpoly):
    point_arr = np.empty((0, 2))
    x = bpoly[0, :]
    y = bpoly[1, :]
    for i in range(len(point_arr)):
        x_point = array[i, 1]
        y_point = array[i, 0]
        y_1 = (y[1, 1] - y[1, 0]) / (x[0, 1] - x[0, 0]) * (x_point - x[0, 1]) + y[
            1, 1]  # y_1 should be smaller than y_point
        y_3 = (y[1, 3] - y[1, 2]) / (x[0, 3] - x[0, 2]) * (x_point - x[0, 3]) + y[1, 3]
        x_2 = (y_point - y[1, 2]) / ((y[1, 2] - y[1, 1]) / (x[0, 2] - x[0, 1])) + x[0, 2]
        x_4 = (y_point - y[1, 0]) / ((y[1, 0] - y[1, 3]) / (x[0, 0] - x[0, 3])) + x[0, 0]

        if (y_point > y_1 and y_point < y_3 and x_point < x_2 and x_point > x_4):
            point_arr = np.vstack((point_arr, array[i]))

    return point_arr