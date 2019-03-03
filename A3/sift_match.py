import cv2
import numpy as np
from fastdtw import fastdtw as dtw
from scipy.spatial.distance import euclidean
# import pdb


def features_keypoints(image, keypoints, window_size):
    """For a given image find all keypoints and their descriptors."""
    kps = [cv2.KeyPoint(x, y, window_size) for x, y in keypoints]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    _, features = descriptor.compute(img, kps)
    return features


def match_greedy_sift(keypoints_A, keypoints_B, features_A, features_B, test_ratio=0.75):
    """Given two sets of key points and their descriptors, find the best matching pairs."""
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    matches = matcher.knnMatch(features_A, features_B, 2)

    refined_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * test_ratio:
            refined_matches.append((m[0].queryIdx, m[0].trainIdx))

    match_A, match_B = [], []
    for idA, idB in refined_matches:
        match_A.append(keypoints_A[idA])
        match_B.append(keypoints_B[idB])

    return np.int32(match_A), np.int32(match_B)


def match_dtw_sift(keypoints_A, keypoints_B, features_A, features_B, test_ratio=0.75):
    """Given two sets of key points and their descriptors, find the best matching pairs."""
    match_A, match_B = [], []

    cost_matrix, path = dtw(features_A, features_B, dist=euclidean)

    for idA, idB in path:
        match_A.append(keypoints_A[idA])
        match_B.append(keypoints_B[idB])

    return np.int32(match_A), np.int32(match_B)


def compute_match_sift(img_A, img_B, window_size=16, stride=8, method="greedy"):
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

    print("Generating features")
    features_A = features_keypoints(img_A, points_A, window_size)
    features_B = features_keypoints(img_B, points_B, window_size)

    print("Matching features")
    if method == "greedy":
        match_A, match_B = match_greedy_sift(points_A, points_B, features_A, features_B)
    elif method == "dtw":
        match_A, match_B = match_dtw_sift(points_A, points_B, features_A, features_B)

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

    cv2.imwrite("output/sift_" + method + "_" + name + ".png", vis)
    cv2.imshow("Keypoint matching", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    method = "dtw"

    image_A = cv2.imread("output/rect_sift_greedy_p1_a.png")
    image_B = cv2.imread("output/rect_sift_greedy_p1_b.png")
    match_A, match_B = compute_match_sift(image_A, image_B, method=method)

    print("Plotting matches")
    plot_images(image_A, image_B, match_A, match_B, name="rect_p1", method=method)
