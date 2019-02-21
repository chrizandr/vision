import cv2
import numpy as np
import pdb


def features_keypoints(image):
    """For a given image find all keypoints and their descriptors."""
    descriptor = cv2.xfeatures2d.SIFT_create()
    kps, features = descriptor.detectAndCompute(image, None)

    keypoints = np.int32([kp.pt for kp in kps])
    return keypoints, features


def match_keypoints(keypoints_A, keypoints_B, features_A, features_B, test_ratio):
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
