import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

from dlt import DLT_method, decompose_projection
from ransac import ransac_estimation
from zhang import zhang_calibration, predict_points_zhang
from wireframe import predict_points, plot_points

if __name__ == "__main__":
    # Use Zhangs method to find image points and world points
    folder = "myimages/"
    image_names = ["img_0.jpg", "img_1.jpg", "img_2.jpg", "img_3.jpg", "img_4.jpg",
                   "img_5.jpg", "img_6.jpg", "img_7.jpg", "img_8.jpg", "img_9.jpg"]
    images = [cv2.imread(os.path.join(folder, x)) for x in image_names]

    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = zhang_calibration(images, size=(15, 10))

    pred_points = predict_points_zhang(rvecs[-1], tvecs[-1], mtx, dist, objpoints[-1])[0].reshape(150, 2)
    img = plot_points(images[-1], pred_points)

    plt.imshow(img)
    plt.show()
    print(mtx)

    image_points = imgpoints[-1].reshape(150, 2).astype(int)
    world_points = objpoints[-1].astype(int)
    projection_matrix = DLT_method(image_points[0:6], world_points[0:6])
    K, C, R = decompose_projection(projection_matrix)
    print(K)
    print(C)
    print(R)
    pred_points = predict_points(projection_matrix, world_points)
    img = plot_points(images[-1], pred_points)

    plt.imshow(img)
    plt.show()

    projection_matrix = ransac_estimation(image_points, world_points)
    K, C, R = decompose_projection(projection_matrix)
    print(K)
    print(C)
    print(R)
    pred_points = predict_points(projection_matrix, world_points)
    img = plot_points(images[-1], pred_points)
    plt.imshow(img)
    plt.show()
