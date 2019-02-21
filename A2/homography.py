import numpy as np
import pdb


def construct_homography(matches_A, matches_B, ransac_iters):
    """Find the best homography given pairs of matching points using RANSAC."""
    n_points = matches_A.shape[0]

    best_P = None
    min_error = np.inf
    for i in range(ransac_iters):
        # print("Iteration number {}".format(i))
        indices = np.random.permutation(n_points)
        X, Y = matches_A[indices[0:4]], matches_B[indices[0:4]]
        X_test, Y_test = matches_A[indices[4::]], matches_B[indices[4::]]

        homography = DLT_method(X, Y)
        res, error = find_projection_error(homography, X_test, Y_test)
        if not res:
            continue

        if error < min_error:
            min_error = error
            best_P = homography

    return best_P, min_error


def DLT_method(matches_A, matches_B):
    assert matches_A.shape[0] == matches_B.shape[0]

    A = np.zeros((matches_A.shape[0]*2, 9))
    i = 0
    for i_points, w_points in zip(matches_A, matches_B):
        u, v = i_points
        x, y = w_points
        A[i] = [0, 0, 0, -u, -v, -1, u*y, v*y, y]
        A[i+1] = [u, v, 1, 0, 0, 0, -u*x, -v*x, -x]
        i += 2

    # Solving using SVD
    u_, D, v_T = np.linalg.svd(A)
    v_T[-1] = v_T[-1]/v_T[-1, -1]
    H = np.reshape(v_T[-1], (3, 3))

    return H


def find_projection_error(H, matches_A, matches_B):
    """Find the error in predicted points for a projection matrix."""
    errors = []

    matches_A = np.pad(matches_A, ((0, 0), (0, 1)), 'constant', constant_values=1)
    matches_B = np.pad(matches_B, ((0, 0), (0, 1)), 'constant', constant_values=1)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return False, 0

    A_ = np.dot(H, matches_A.T).T
    A_ = A_[:, :2] / A_[:, -1].reshape(matches_A.shape[0], 1)
    d1 = np.linalg.norm(A_[:, :2] - matches_B[:, :2], ord=2, axis=1)

    B_ = np.dot(H_inv, matches_B.T).T
    B_ = B_ / B_[:, -1].reshape(matches_B.shape[0], 1)
    d2 = np.linalg.norm(B_[:, :2] - matches_A[:, :2], ord=2, axis=1)

    errors = d1 + d2
    outliers = np.nonzero(errors > 0.5)[0]

    return True, len(outliers)
