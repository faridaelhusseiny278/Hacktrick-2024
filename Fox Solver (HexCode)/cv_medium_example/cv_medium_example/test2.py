import cv2
import numpy as np


def remove_patch(base_image, patch_image):
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)  # b3ml el image gray
    patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)  # b3ml el images gray
    sift = cv2.SIFT.create()  # this is use intialize sift

    kp1, des1 = sift.detectAndCompute(base_gray, None)
    kp2, des2 = sift.detectAndCompute(patch_gray, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.90 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)
        mask_warped = cv2.warpPerspective(np.ones_like(patch_gray) * 255, M, (base_image.shape[1], base_image.shape[0]))
        base_image_without_patch = cv2.inpaint(base_image, mask_warped.astype(np.uint8), inpaintRadius=20,
                                               flags=cv2.INPAINT_TELEA)

        return base_image_without_patch


if __name__ == "__main__":
    base_image = cv2.imread('combined_large_image.png')
    patch_image = cv2.imread('patch_image.png')
    base_image_removed_patch = remove_patch(base_image, patch_image)

    cv2.imshow('Base Image with Patch Removed', base_image_removed_patch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
