import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from wireframe import plot_points


def find_img_points(img, size=(8, 6)):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((size[0]*size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return objp, corners2, gray.shape
    return None, None, None


def zhang_calibration(images, size):
    objpoints = []
    imgpoints = []
    for i, img in enumerate(images):
        print("Processing image ", i)
        objp, imgp, shape = find_img_points(img, size)
        if objp is not None and imgp is not None:
            objpoints.append(objp)
            imgpoints.append(imgp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints


def predict_points_zhang(rvecs, tvecs, mtx, dist, world_points):
    img_points = cv2.projectPoints(world_points, rvecs, tvecs, mtx, dist)
    return img_points


if __name__ == "__main__":
    folder = "Assignment1_Data/"
    image_names = ["IMG_5456.JPG", "IMG_5457.JPG", "IMG_5458.JPG", "IMG_5459.JPG", "IMG_5460.JPG",
                   "IMG_5461.JPG", "IMG_5462.JPG", "IMG_5463.JPG", "IMG_5464.JPG", "IMG_5465.JPG",
                   "IMG_5466.JPG", "IMG_5467.JPG", "IMG_5468.JPG", "IMG_5469.JPG", "IMG_5470.JPG"]
    images = [cv2.imread(os.path.join(folder, x)) for x in image_names]

    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = zhang_calibration(images, size=(8, 6))
    pred_points = predict_points_zhang(rvecs[-1], tvecs[-1], mtx, dist, objpoints[-1])[0].reshape(48, 2)
    img = plot_points(images[-1], pred_points)

    plt.imshow(img)
    plt.show()
    print(mtx)
