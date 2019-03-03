import cv2
import numpy as np
import pdb

from sift_match import compute_match_sift
from window_match import compute_match_window


def rectify_images_sift(image_A, image_B, window_size=16, stride=8, method="greedy", name="p1"):
    """Rectify two stereo images."""
    print("Finding matching points")
    match_A, match_B = compute_match_sift(image_A, image_B, method=method)

    print("Finding Fundamantel Matrix")
    F, mask = cv2.findFundamentalMat(match_A, match_B)

    print("Computing homography")
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(match_A, match_B, F, image_A.shape[0:2])

    print("Rectifying images")
    new_img_A = cv2.warpPerspective(image_A, H1, image_A.shape[0:2])
    new_img_B = cv2.warpPerspective(image_B, H2, image_A.shape[0:2])

    cv2.imwrite("output/rect_sift_" + method + "_" + name + "_a" + ".png", new_img_A)
    cv2.imwrite("output/rect_sift_" + method + "_" + name + "_b" + ".png", new_img_B)

    return new_img_A, new_img_B


def rectify_images_window(image_A, image_B, window_size=30, stride=30, method="greedy", name="p1"):
    """Rectify two stereo images."""
    print("Finding matching points")
    match_A, match_B = compute_match_window(image_A, image_B, method=method)

    print("Finding Fundamantel Matrix")
    F, mask = cv2.findFundamentalMat(match_A, match_B)

    print("Computing homography")
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(match_A, match_B, F, image_A.shape[0:2])

    print("Rectifying images")
    new_img_A = cv2.warpPerspective(image_A, H1, image_A.shape[0:2])
    new_img_B = cv2.warpPerspective(image_B, H2, image_A.shape[0:2])

    cv2.imwrite("output/rect_window_" + method + "_" + name + "_a" + ".png", new_img_A)
    cv2.imwrite("output/rect_window_" + method + "_" + name + "_b" + ".png", new_img_B)

    return new_img_A, new_img_B


def show_rectified(img_A, img_B, name="p1", method="sift"):
    """Show the images side by side after rectification."""
    (hA, wA) = img_A.shape[:2]
    (hB, wB) = img_B.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = img_A
    vis[0:hB, wA:] = img_B

    cv2.imwrite("output/rect_" + method + "_" + name + ".png", vis)
    cv2.imshow("Keypoint matching", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    image_A = cv2.imread("Assignment_Data/p3_a.png")
    image_B = cv2.imread("Assignment_Data/p3_b.png")
    new_img_A, new_img_B = rectify_images_sift(image_A, image_B, method="greedy", name="p3")
    show_rectified(new_img_A, new_img_B, name="p3", method="sift")

    image_A = cv2.imread("Assignment_Data/p3_a.png")
    image_B = cv2.imread("Assignment_Data/p3_b.png")
    new_img_A, new_img_B = rectify_images_window(image_A, image_B, method="greedy", name="p3")
    show_rectified(new_img_A, new_img_B, name="p3", method="window")
