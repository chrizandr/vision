import numpy as np
import pdb
from scipy import linalg


def decompose_projection(P):
    """Decompose the projection matrix into K, R, C"""
    M = P[:, 0:3]
    MC = -1 * P[:, 3]
    C = np.dot(np.linalg.inv(M), MC)
    K, R = linalg.rq(M)
    return K, R, C


def reconstruct_projection(K, R, C):
    """Create the projection matrix to be multiplied in camera equation."""
    M = np.dot(K, R)
    MC = -1 * np.dot(M, C)
    P = np.hstack((M, MC))
    return P


def DLT_method(image_points, world_points):
    assert image_points.shape[0] == world_points.shape[0]

    A = np.zeros((image_points.shape[0]*2, 11))
    U = np.zeros((image_points.shape[0]*2, 1))
    i = 0
    for i_points, w_points in zip(image_points, world_points):
        u, v = i_points
        x, y, z = w_points
        A[i] = [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z]
        U[i] = -u
        A[i+1] = [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z]
        U[i+1] = -v

        i += 2

    # Solving using SVD
    H = np.hstack((A, U))
    u_, D, v_T = np.linalg.svd(H)
    P = (v_T[-1]).reshape((-1, 1))
    divider = P[-1, 0]
    P = P/divider

    P = P.reshape(3, 4)

    return P


def find_projection_error(P, image_points, world_points):
    """Find the error in predicted points for a projection matrix."""
    total_error = 0
    for img_point, (x, y, z) in zip(image_points, world_points):
        A = np.array([[x], [y], [z], [1]])
        i_pred = np.dot(P, A)
        i_pred = i_pred.reshape(1, -1)[0]
        i_pred = i_pred[:-1] / i_pred[-1]
        error = np.linalg.norm(i_pred - img_point, ord=2)

        total_error += error

    return total_error


if __name__ == "__main__":
    world_points = np.array([[36, 0, 0], [0, 0, 36], [0, 36, 0],
                             [36, 36, 0], [36, 0, 36], [0, 0, 72]])
    image_points = np.array([[396.505, 209.674], [473.951, 246.394], [486.636, 138.904],
                             [402.514, 132.227], [385.822, 237.047], [465.94, 277.77]])
    projection_matrix = DLT_method(image_points, world_points)
    K, R, C = decompose_projection(projection_matrix)
    print(K)
    print(R)
    print(C)
    error = find_projection_error(projection_matrix, image_points, world_points)
    print(error)
