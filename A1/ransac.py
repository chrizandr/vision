import numpy as np
# import pdb

from dlt import DLT_method, find_projection_error, decompose_projection


def ransac_estimation(image_points, world_points, iter=500):
    """Use the RANSAC algorithm with DLT to find best projection matrix"""
    n_points = image_points.shape[0]
    best_P = None
    min_error = np.inf
    for i in range(iter):
        print("Iteration number {}".format(i))
        indices = np.random.permutation(n_points)
        X, Y = image_points[indices[0:6]], world_points[indices[0:6]]
        X_test, Y_test = image_points[indices[6::]], world_points[indices[6::]]

        projection_matrix = DLT_method(X, Y)
        error = find_projection_error(projection_matrix, X_test, Y_test)

        if error < min_error:
            min_error = error
            best_P = projection_matrix

    return best_P


if __name__ == "__main__":
    from points import image_points, world_points
    projection_matrix = ransac_estimation(image_points, world_points)
    K, R, C = decompose_projection(projection_matrix)
    print(K)
    print(R)
    print(C)
