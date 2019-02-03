import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import linalg


def find_corners(img):
    """Harris corner detection to find the points in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dist = cv2.cornerHarris(gray, 2, 3, 0.04)
    dist = cv2.dilate(dist, None)
    ret, dist = cv2.threshold(dist, 0.01*dist.max(), 255, 0)
    dist = np.uint8(dist)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dist)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    # img[res[:,1],res[:,0]]=[0,0,255]
    for i in range(res.shape[0]):
        x, y = res[i, 3], res[i, 2]
        img[x-2:x+2, y-2:y+2] = [0, 255, 0]

    plt.imshow(img)
    plt.show()
    pdb.set_trace()


def ransac_camera_calibrate(image_points, world_points):
    N = image_points.shape[0]
    n = 100
    min_error = 10**8
    opt_P = np.zeros((3, 4))
    opt_i = -1
    for i in range(n):
        m = np.random.choice(N, 6, replace=False)
        i_points = image_points[m]
        w_points = world_points[m]
        P = camera_calibrate(i_points, w_points)
        m_ = np.array(list(set(range(N)) - set(m)))
        test_i_points = image_points[m_]
        test_w_points = world_points[m_]
        test_w_points = np.concatenate([test_w_points, np.ones((len(m_), 1))], axis=-1)
        pdb.set_trace()
        predicted_i_points = np.dot(P, test_w_points.T).T
        predicted_i_points = predicted_i_points/predicted_i_points[:, -1].reshape(len(m_), 1)
        error = ((predicted_i_points[:, :2] - test_i_points)**2).sum()/len(m_)
        if error < min_error:
            min_error = error
            opt_P = P
            opt_i = i

    print(min_error, opt_P, opt_i)
    return opt_P


if __name__ == "__main__":
    img = cv2.imread('Assignment1_Data/measurements.jpg')
    world_points = [[36, 0, 0], [0, 0, 36], [0, 36, 0],
                    [36, 36, 0], [36, 0, 36], [0, 0, 72]]
    image_points = [[396.505, 209.674], [473.951, 246.394], [486.636, 138.904],
                    [402.514, 132.227], [385.822, 237.047], [465.94, 277.77]]
    P = camera_calibrate(image_points, world_points)
    K, R, C = decompose_projection(P)

    image_points = []
    world_points = []
    with open('points.txt') as f:
        for line in f:
            line = line.strip().split(',')
            world_points.append([float(i.strip()) for i in line[0:3]])
            image_points.append([float(i.strip()) for i in line[3:]])
    image_points = np.array(image_points)
    world_points = np.array(world_points)
    ransac_camera_calibrate(image_points, world_points)
    # find_corners(img)
