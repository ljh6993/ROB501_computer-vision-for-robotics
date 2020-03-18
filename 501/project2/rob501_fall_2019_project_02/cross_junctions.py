# import picture
# harris corner
# convolution
# show the result in the bound
# show plt


import numpy as np
from scipy.ndimage.filters import *


# use at last

def average_point(sorted_p):
    distance = 10
    # sort column 0 then column 1
    # idex = np.lexsort([ori_point[:, 1], ori_point[:, 0]])

    # print(sorted_p)

    feature_list = np.empty([0, 2])
    feature_list = np.vstack([feature_list, sorted_p[0]])  # total 48 feature from 0-47
    # feature_list numpy
    feature_data = 0
    tf = False
    data_list = []
    for i in range(len(sorted_p)):
        reference = sorted_p[i]
        for i in range(len(feature_list)):
            if np.linalg.norm(feature_list[i] - reference) < distance:
                # np.array_equal(feature_list[i],reference):
                tf = True
                feature_data = i
                break
            tf = False
            feature_data = i + 1

        if tf == True:
            data_list.append(feature_data)
        else:
            feature_list = np.vstack([feature_list, reference])
            data_list.append(feature_data)
    data_list = np.array(data_list).reshape([-1, 1])
    sorted_p = np.hstack((sorted_p, data_list))

    position = []
    for i in range(48):
        position.append(np.mean(sorted_p[np.where(sorted_p[:, 2] == i)], axis=0))
    position = np.array(position)
    position = position[:, [0, 1]]

    return (position)


# part 1.5
# check cross color
def filter_p(ori_point, I_bin, width, height):
    new_point = []
    detect_size = 3

    for i in range(len(ori_point)):
        x = ori_point[i, 1]
        y = ori_point[i, 0]
        x = int(x)
        y = int(y)
        if x in range(detect_size, width - detect_size) and y in range(detect_size, height - detect_size):
            p_1 = I_bin[y - detect_size, x - detect_size]
            p_2 = I_bin[y - detect_size, x + detect_size]
            p_3 = I_bin[y + detect_size, x + detect_size]
            p_4 = I_bin[y + detect_size, x - detect_size]
            p_5 = I_bin[y - detect_size, x]
            p_6 = I_bin[y, x + detect_size]
            p_7 = I_bin[y + detect_size, x]
            p_8 = I_bin[y, x - detect_size]
            tf_1 = ((p_1 == p_3) and (p_2 == p_4) and (p_2 * p_4) != (p_1 * p_3))
            tf_2 = ((p_5 == p_7) and (p_6 == p_8) and (p_5 * p_7) != (p_6 * p_8))

            if tf_1 or tf_2:
                new_point.append(ori_point[i])

    new_point = np.array(new_point)

    return new_point


def saddle(filter_point, I):
    ###implement saddle point
    # part 2
    # build a small patch for each point
    patch_size = 20  # need to be even
    sdpoint_arr = np.empty((0, 2))
    for i in range(len(filter_point)):
        # saddle point position
        sdpoint_x = filter_point[i][1]
        sdpoint_y = filter_point[i][0]

        #     a1=sdpoint_x-patch_size/2
        #     a2=sdpoint_x+patch_size/2
        #     a3=sdpoint_y-patch_size/2
        #     a4=sdpoint_y+patch_size/2
        #     print(a1,a2,a3,a4)
        #  do the saddle point based on I_binary
        patch = I[int(sdpoint_y - patch_size / 2):int(sdpoint_y + patch_size / 2),
                int(sdpoint_x - patch_size / 2):int(sdpoint_x + patch_size / 2)]
        m, n = patch.shape
        # clear B
        B = []
        # x2 xy y2 x y 1 (m,n) begin from 0,0
        A = np.empty((0, 6))
        for x in range(0, n):
            for y in range(0, m):
                A = np.vstack((A, [x ** 2, x * y, y ** 2, x, y, 1.]))
                B.append(patch[y, x])
        # w=(a,b,c,d,e,f) for parameters
        w = np.linalg.lstsq(A, B, rcond=None)[0]
        (a, b, c, d, e, f) = w
        # solve the intersection
        # [2a b],[b 2c]
        C = np.array([[float(2 * a), float(b)], [float(b), float(2 * c)]])
        D = np.array([-d, -e])
        pt = np.linalg.pinv(C).dot(D)
        # correct the position of pt
        # print(pt.shape)
        pt = pt - [patch_size / 2, patch_size / 2] + [sdpoint_x, sdpoint_y]
        pt = pt.reshape((1, 2))
        # print(pt)(x,y)
        sdpoint_arr = np.append(sdpoint_arr, pt, axis=0)

        # print(sdpoint_arr.shape) #(n,2)
    sdpoint_arr = sdpoint_arr[:, [1, 0]]
    return sdpoint_arr


def boundary_delete(array, bpoly):
    point_arr = np.empty((0, 2))
    x = np.array(bpoly[0, :])
    y = np.array(bpoly[1, :])
    # print(len(point_arr)
    for i in range(len(array)):
        x_point = array[i, 1]
        y_point = array[i, 0]
        y_1 = (y[1] - y[0]) / (x[1] - x[0]) * (x_point - x[1]) + y[1] + 5  # y_1 should be smaller than y_point
        y_3 = (y[3] - y[2]) / (x[3] - x[2]) * (x_point - x[3]) + y[3] - 5
        x_2 = (y_point - y[2]) / ((y[2] - y[1]) / (x[2] - x[1])) + x[2] - 5
        x_4 = (y_point - y[0]) / ((y[0] - y[3]) / (x[0] - x[3])) + x[0] + 5

        #         y_1 = (y[1, 1] - y[1, 0]) / (x[0, 1] - x[0, 0]) * (x_point - x[0, 1]) + y[1, 1]  # y_1 should be smaller than y_point
        #         y_3 = (y[1, 3] - y[1, 2]) / (x[0, 3] - x[0, 2]) * (x_point - x[0, 3]) + y[1, 3]
        #         x_2 = (y_point - y[1, 2]) / ((y[1, 2] - y[1, 1]) / (x[0, 2] - x[0, 1])) + x[0, 2]
        #         x_4 = (y_point - y[1, 0]) / ((y[1, 0] - y[1, 3]) / (x[0, 0] - x[0, 3])) + x[0, 0]

        if (y_point > y_1 and y_point < y_3 and x_point < x_2 and x_point > x_4):
            point_arr = np.vstack((point_arr, array[i]))

    return point_arr


from scipy.ndimage.filters import *


def cross_junctions(I, bounds, Wpts):
    # part one
    # change to binary
    I_binary = np.where(I > 115, 0, 1)
    # gaussian_filter filter
    I_gaussian = gaussian_filter(I, sigma=1)

    I_copy = I.copy()

    # harris corner
    # calculate gradient

    Ix, Iy = np.gradient(I_binary)
    Ix = gaussian_filter(Ix, sigma=1)
    Iy = gaussian_filter(Iy, sigma=1)

    # 2d array
    Ixx = Ix ** 2
    Ixy = Iy * Ix
    Iyy = Iy ** 2
    # Ixx=gaussian_filter(Ixx, sigma=2)
    # Ixy=gaussian_filter(Ixy, sigma=2)
    # Iyy=gaussian_filter(Iyy, sigma=2)
    height = I_binary.shape[0]  # 480
    width = I_binary.shape[1]  # 640

    k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    Sxx = convolve(Ixx, k, mode='nearest')
    Sxy = convolve(Ixy, k, mode='nearest')
    Syy = convolve(Iyy, k, mode='nearest')

    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy
    r = det - 0.04 * (trace ** 2)

    # array_np = np.array(r)
    low_values_flags = r < 0  # Where values are low
    r[low_values_flags] = 0

    # max=np.amax(r)
    # print(max)

    # print(array_np)(y,x)
    point_array = []
    ## show the plot
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if r[i, j] != 0:
                # I_copy[i,j]=255
                point_array.append([i, j])
    # point_array is integer listxxxx turn to be a numpy array
    point_array = np.array(point_array)

    # print(point_array)
    # plt.plot(point_array[:,1],point_array[:,0],'ro', color='r')

    # filter point (y,x)
    # filter_point=filter_p(point_array,I_binary,width,height)

    # run the function
    # sdpoint(y,x)

    sdpoint = np.empty((0, 2))

    sdpoint = saddle(point_array, I_gaussian)

    sdpoint = boundary_delete(sdpoint, bounds)
    sdpoint = filter_p(sdpoint, I_binary, width, height)

    # np.set_printoptions(threshold=np.inf)
    # sdpoint

    # need to sort point !!
    # print(sorted_point)

    final_result = average_point(sdpoint)
    #   Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
    #   left corner of I. These should be floating-point values.
    # $
    idex = np.lexsort([final_result[:, 1], final_result[:, 0]])
    final_result = final_result[idex, :]
    order = np.array([])
    for i in range(6):
        ordd=np.argsort(final_result[i*8:(i+1)*8,1])
        ordd = ordd + 8 * i
        order = np.append(order, ordd)

    final_result = final_result[order.astype(int)]


    Ipts = np.zeros((2, 48))
    Ipts = final_result.transpose()[[1, 0]]

    # ------------------
    return Ipts



