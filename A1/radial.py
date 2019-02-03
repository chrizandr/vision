import cv2
import matplotlib.pyplot as plt
import numpy as np
import pdb

def remove_distortion(calib_img, img_points, world_points, distort_img):
    """Removes the distortion from an image by finding camera properties."""
    gray = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
    # pdb.set_trace()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array([world_points]), np.array([img_points]),
                                                       gray.shape[::-1], None, None)
    h,  w = distort_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(distort_img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst


if __name__ == "__main__":
    from points import world_points, image_points
    calib_img = cv2.imread('Assignment1_Data/IMG_5455.JPG')

    distort_img = cv2.imread('Assignment1_Data/IMG_5455.JPG')
    dst = remove_distortion(calib_img, image_points, world_points, distort_img)

    plt.imshow(distort_img)
    plt.show()

    plt.imshow(dst)
    plt.show()
