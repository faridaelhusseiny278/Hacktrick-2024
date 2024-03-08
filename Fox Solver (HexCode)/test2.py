import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
def remove_patch(base_image, patch_image):

    # Convert the list to a NumPy array
    base_image = np.array(base_image, dtype=np.uint8)
    patch_image = np.array(patch_image, dtype=np.uint8)
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)  # b3ml el image gray
    patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)  # b3ml el images gray
    sift = cv2.SIFT.create()  # this is use initialize sift

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



        return base_image_without_patch.tolist()


#if __name__ == "__main__":
#    base_image = Image.open("cv_medium_example/cv_medium_example/combined_large_image.jpg")
#    base_img_list = list(base_image.getdata())
#    width, height = base_image.size
#    base_img_3d_list = [base_img_list[i * width:(i + 1) * width] for i in range(height)]
#
#    patch_image = Image.open("cv_medium_example/cv_medium_example/patch_image.jpg")
#    patch_img_list = list(patch_image.getdata())
#    width, height = base_image.size
#    patch_img_3d_list = [patch_img_list[i * width:(i + 1) * width] for i in range(height)]
#
#    base_image_removed_patch = remove_patch(base_image, patch_image)
#    # Reshape the 2D list to a 3D list

