import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

from dlt import DLT_method
from ransac import ransac_estimation


def predict_points(projection_matrix, world_points):
    """Use the projection matrix to predict image_points"""
    points = []
    for (x, y, z) in world_points:
        A = np.array([[x], [y], [z], [1]])
        i_pred = np.dot(projection_matrix, A)
        i_pred = i_pred.reshape(1, -1)[0]
        i_pred = i_pred[:-1] / i_pred[-1]
        points.append(i_pred)

    points = np.array(points, dtype=np.int)
    return points


def plot_wireframe(image, points):
    img = image.copy()
    for (x, y) in points:
        img = cv2.circle(img, (x, y), 50, (255, 0, 0), -1)

    for p in points:
        distance = np.sum((points - p)**2, axis=1)
        min_index = np.argsort(distance)[1:5]
        for nearest_point in points[min_index]:
            img = cv2.line(img, tuple(p), tuple(nearest_point), (255, 0, 0), 20)
    return img


if __name__ == "__main__":
    from points import image_points, world_points

    image = cv2.imread('Assignment1_Data/IMG_5455.JPG')
    projection_matrix = DLT_method(image_points, world_points)

    pred_points = predict_points(projection_matrix[0:6], world_points[0:6])
    img = plot_wireframe(image, pred_points)

    plt.imshow(img)
    plt.show()

    projection_matrix = ransac_estimation(image_points, world_points)

    pred_points = predict_points(projection_matrix, world_points)
    img = plot_wireframe(image, pred_points)

    plt.imshow(img)
    plt.show()
