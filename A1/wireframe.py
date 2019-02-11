import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

from dlt import DLT_method
from ransac import ransac_estimation
# from zhang import zhang_calibration, predict_points_zhang


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


def plot_points(image, points):
    img = image.copy()
    for (x, y) in points:
        if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
            continue
        img = cv2.circle(img, (x, y), 50, (255, 0, 0), -1)
    return img


if __name__ == "__main__":
    from points import image_points, world_points

    image = cv2.imread('Assignment1_Data/IMG_5455.JPG')
    projection_matrix = DLT_method(image_points, world_points)

    pred_points = predict_points(projection_matrix, world_points[0:6])
    img = plot_points(image, pred_points)

    plt.imshow(img)
    plt.show()

    projection_matrix = ransac_estimation(image_points, world_points)
    pred_points = predict_points(projection_matrix, world_points)
    img = plot_points(image, pred_points)
    plt.imshow(img)
    plt.show()

    folder = "Assignment1_Data/"
    image_names = ["IMG_5456.JPG", "IMG_5457.JPG", "IMG_5458.JPG", "IMG_5459.JPG", "IMG_5460.JPG",
                   "IMG_5461.JPG", "IMG_5462.JPG", "IMG_5463.JPG", "IMG_5464.JPG", "IMG_5465.JPG",
                   "IMG_5466.JPG", "IMG_5467.JPG", "IMG_5468.JPG", "IMG_5469.JPG", "IMG_5470.JPG"]
    images = [cv2.imread(os.path.join(folder, x)) for x in image_names]

    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = zhang_calibration(images, size=(8, 6))
    pred_points = predict_points_zhang(rvecs[-1], tvecs[-1], mtx, dist, objpoints[-1])[0].reshape(48, 2)
    img = plot_wireframe(images[-1], pred_points)

    plt.imshow(img)
    plt.show()
