import cv2
import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
# import pdb


def correlate(a, b):
    """For a given pair of images find the correlation between them"""
    assert a.shape == b.shape
    a = a - a.mean()
    b = b - b.mean()
    a_sum = np.sum(a**2)
    b_sum = np.sum(b**2)

    numerator = np.sum(a * b)
    denominator = np.sqrt(a_sum * b_sum)

    return numerator/denominator


def match_greedy_window(keypoints_A, keypoints_B, img_A, img_B, window_size):
    """Given two sets of key points and their descriptors, find the best matching pairs."""
    gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
    gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

    dist = np.zeros((len(keypoints_A), len(keypoints_B)), dtype=np.float)

    back = window_size // 2
    front = window_size // 2 - 1
    num_matches = float(len(keypoints_A) * len(keypoints_B))
    print(num_matches)

    for i, kA in tqdm(enumerate(keypoints_A)):
        for j, kB in tqdm(enumerate(keypoints_B)):
            window_A = gray_A[kA[1]-back:kA[1]+front, kA[0]-back:kA[0]+front]
            window_B = gray_B[kB[1]-back:kB[1]+front, kB[0]-back:kB[0]+front]
            dist[i, j] = correlate(window_A, window_B)

    match_A = keypoints_A
    match_B = [keypoints_B[np.argmax(dist[i])] for i in range(len(keypoints_A))]

    return np.int32(match_A), np.int32(match_B)


def match_dtw_window(keypoints_A, keypoints_B, img_A, img_B, window_size):
    """Given two sets of key points and their descriptors, find the best matching pairs."""
    gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
    gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

    dist = np.zeros((len(keypoints_A), len(keypoints_B)), dtype=np.float)

    back = window_size // 2
    front = window_size // 2 - 1
    num_matches = float(len(keypoints_A) * len(keypoints_B))
    print(num_matches)

    for i, kA in tqdm(enumerate(keypoints_A)):
        for j, kB in tqdm(enumerate(keypoints_B)):
            window_A = gray_A[kA[1]-back:kA[1]+front, kA[0]-back:kA[0]+front]
            window_B = gray_B[kB[1]-back:kB[1]+front, kB[0]-back:kB[0]+front]
            dist[i, j] = 1 - correlate(window_A, window_B)

    def distance(i, j):
        return dist[i, j]

    x = np.arange(len(keypoints_A)).reshape(-1, 1)
    y = np.arange(len(keypoints_B)).reshape(-1, 1)

    _, _, _, (idxA, idxB) = dtw(x, y, dist=distance)

    match_A = []
    match_B = []

    for idA, idB in zip(idxA, idxB):
        match_A.append(keypoints_A[idA])
        match_B.append(keypoints_B[idB])

    return np.int32(match_A), np.int32(match_B)


def compute_match_window(img_A, img_B, window_size=50, stride=50, method="greedy"):
    """Compute the matching points between two images using dense SIFT."""
    m1, n1 = img_A.shape[0:2]
    m2, n2 = img_B.shape[0:2]

    print("Computing points")
    points_A = []
    for y in range(window_size, m1 - window_size, stride):
        for x in range(window_size, n1 - window_size, stride):
            points_A.append((x, y))

    points_B = []
    for y in range(window_size, m2 - window_size, stride):
        for x in range(window_size, n2 - window_size, stride):
            points_B.append((x, y))

    print("Matching windows")
    if method == "greedy":
        match_A, match_B = match_greedy_window(points_A, points_B, img_A, img_B, window_size)
    elif method == "dtw":
        match_A, match_B = match_dtw_window(points_A, points_B, img_A, img_B, window_size)

    return match_A, match_B


def plot_images(img_A, img_B, match_A, match_B, name="p1", method="greedy"):
    """Plot the images and the matching keypoints."""
    (hA, wA) = img_A.shape[:2]
    (hB, wB) = img_B.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = img_A
    vis[0:hB, wA:] = img_B

    # loop over the matches
    for ptA, ptB in zip(match_A, match_B):
        ptB = ptB + np.array([wA, 0])
        cv2.line(vis, tuple(ptA), tuple(ptB), (0, 255, 0), 1)

    cv2.imwrite("output/window_" + method + "_" + name + ".png", vis)
    cv2.imshow("Keypoint matching", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    method = "dtw"

    image_A = cv2.imread("Assignment_Data/p1_a.png")
    image_B = cv2.imread("Assignment_Data/p1_b.png")
    match_A, match_B = compute_match_window(image_A, image_B, method=method)

    print("Plotting matches")
    plot_images(image_A, image_B, match_A, match_B, name="p1", method=method)
