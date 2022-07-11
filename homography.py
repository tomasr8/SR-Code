import math
import numpy as np
import cv2

def estimate_homography(image_points, world_points, method=cv2.RANSAC):
    """Estimate homography from image-world correspondences.

    Args:
        image_points: Nx2 numpy array of image points.
        world_points: Nx2 numpy array of world points.

    Returns:
        The normalized homography matrix.
    """

    # method=0 uses least-squares
    # other methods are:
    #   - RANSAC (use method=cv2.RANSAC)
    #   - Least-Median (method=cv2.LMEDS)
    #   - PROSAC-based method (method=cv2.RHO)
    #
    # See docs: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    #
    H, _ = cv2.findHomography(image_points, world_points, method=method)

    return H



def solve_homography(A, B):
    '''
        Taken from: https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    '''
    [[a1x, a1y], [a2x, a2y], [a3x, a3y], [a4x, a4y]] = A
    [[b1x, b1y], [b2x, b2y], [b3x, b3y], [b4x, b4y]] = B

    P = np.array([
        [-a1x, -a1y, -1, 0, 0, 0, a1x*b1x, a1y*b1x, b1x],
        [0, 0, 0, -a1x, -a1y, -1, a1x*b1y, a1y*b1y, b1y],

        [-a2x, -a2y, -1, 0, 0, 0, a2x*b2x, a2y*b2x, b2x],
        [0, 0, 0, -a2x, -a2y, -1, a2x*b2y, a2y*b2y, b2y],

        [-a3x, -a3y, -1, 0, 0, 0, a3x*b3x, a3y*b3x, b3x],
        [0, 0, 0, -a3x, -a3y, -1, a3x*b3y, a3y*b3y, b3y],

        [-a4x, -a4y, -1, 0, 0, 0, a4x*b4x, a4y*b4x, b4x],
        [0, 0, 0, -a4x, -a4y, -1, a4x*b4y, a4y*b4y, b4y],

        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.float)

    b = np.zeros(9, dtype=np.float)
    b[-1] = 1

    H = np.linalg.solve(P, b)

    H = H.reshape(3, 3)

    return H

if __name__ == "__main__":
    A = np.array([
        [467, 683],
        [819, 680],
        [775, 570],
        [519, 573]
    ], dtype=np.float)

    B = np.array([
        [0, 0],
        [80, 0],
        [80, 80],
        [0, 80]
    ], dtype=np.float)

    H = solve_homography(A, B)
    H /= H[2, 2]
    print(H.round(2))

    x = np.array([880, 449, 1], dtype=np.float).reshape(3, 1)

    y = H @ x
    y /= y[2]
    print(y[:2])
