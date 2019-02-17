import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb

from homography import construct_homography
from keypoints import features_keypoints, match_keypoints


def stitch_multiple(images):
    """Stitch multiple images together."""
    res = images[0]
    i = 0
    for img in images[1:]:
        print(">>> Stitch image ", i, " <<<")
        res = stitch(res, img)
        i += 1
    return res


def stitch(imageA, imageB, test_ratio=0.75, ransac_iters=5000):
    """Stitch two given images into a single large image."""
    print("Finding keypoints")
    key_points_A, features_A = features_keypoints(imageA)
    key_points_B, features_B = features_keypoints(imageB)

    print("Matching keypoints")
    matches_A, matches_B = match_keypoints(key_points_A, key_points_B,
                                           features_A, features_B, test_ratio)
    if matches_A.shape[0] < 4:
        raise ValueError("The given images cannot be matched")

    # show_matches(imageA, imageB, matches_A, matches_B)
    print("Finding homography")
    homography, error = construct_homography(matches_A, matches_B, ransac_iters)

    print("Warping and combining")
    stitched_image = warp_project(imageB, imageA, homography)
    # pdb.set_trace()

    return stitched_image


def show_matches(imageA, imageB, matches_A, matches_B):
    """Plot the images and the matching keypoints."""
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for ptA, ptB in zip(matches_A, matches_B):
        ptB = ptB + np.array([wA, 0])
        cv2.line(vis, tuple(ptA), tuple(ptB), (0, 255, 0), 1)

    cv2.imshow("Keypoint matching", vis)
    cv2.waitKey(0)


def warp_project(imageA, imageB, homography):
    """Join two images given the homography matrix."""
    # hA, wA = imageA.shape[:2]
    # hB, wB = imageB.shape[:2]
    #
    # final_image = cv2.warpPerspective(imageA, homography, (wA+wB, hA+hB))
    # pdb.set_trace()
    # final_image[0:hB, 0:wB] = imageB
    #
    # cv2.imwrite("res_2.png", final_image)
    #
    # return final_image
    h1, w1 = imageA.shape[:2]
    h2, w2 = imageB.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, homography)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(imageB, Ht.dot(homography), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = imageA
    return result

    # cv2.waitKey(0)


def mix_and_match(base_img, warped_img):
    i1y, i1x = base_img.shape[:2]
    i2y, i2x = warped_img.shape[:2]

    for i in range(0, i1x):
        for j in range(0, i1y):
            try:
                if(np.array_equal(base_img[j, i], np.array([0, 0, 0])) and
                   np.array_equal(warped_img[j, i], np.array([0, 0, 0]))):
                    warped_img[j, i] = [0, 0, 0]
                else:
                    if(np.array_equal(warped_img[j, i], [0, 0, 0])):
                        warped_img[j, i] = base_img[j, i]
                    else:
                        if not np.array_equal(base_img[j, i], [0, 0, 0]):
                            bl, gl, rl = base_img[j, i]
                            warped_img[j, i] = [bl, gl, rl]
            except:
                pass

    return warped_img


if __name__ == "__main__":
    # imageA = cv2.imread("Assignment_Data/img1_1.png")
    # imageB = cv2.imread("Assignment_Data/img1_2.png")
    #
    # stitched_image = stitch(imageB, imageA)

    # images = ["img2_1.png", "img2_2.png", "img2_3.png", "img2_4.png", "img2_5.png", "img2_6.png"]
    # images = [cv2.imread("Assignment_Data/" + x) for x in images]
    #
    # stitched_image = stitch_multiple(images)
    #
    # cv2.imwrite("res_2.png", stitched_image)
    #
    # images = ["img3_1.png", "img3_2.png"]
    # images = [cv2.imread("Assignment_Data/" + x) for x in images]
    #
    # stitched_image = stitch_multiple(images)
    #
    # cv2.imwrite("res_3.png", stitched_image)

    images = ["img4_1.jpg", "img4_2.jpg"]
    images = [cv2.imread("Assignment_Data/" + x) for x in images]

    stitched_image = stitch_multiple(images)

    cv2.imwrite("res_3.png", stitched_image)

    plt.imshow(stitched_image)
    plt.show()
