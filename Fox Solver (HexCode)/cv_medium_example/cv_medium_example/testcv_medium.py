import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import random
from matplotlib import rcParams


def remove_patch(base_image, patch_image):
    # Convert images to grayscale
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    patch_gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY)

    # Resize patch image to match size of the patch in the base image
    patch_height, patch_width = patch_gray.shape[:2]
    # patch_image_resized = cv2.resize(patch_gray, (50, 50))
    # Perform template matching
    result = cv2.matchTemplate(base_gray, patch_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # # Extract the coordinates of the top-left and bottom-right corners of the patch
    top_left = max_loc
    bottom_right = (top_left[0] + patch_width, top_left[1] + patch_height)

    base_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255

    return base_image


def interpolate_missing_pixels(base_image):
    # Convert the base image to grayscale
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    # Create a mask for missing pixels (where the pixel values are white)
    mask = cv2.threshold(base_gray, 250, 255, cv2.THRESH_BINARY)[1]

    # Inpaint missing pixels
    base_image1 = cv2.inpaint(base_image, mask, inpaintRadius=20, flags=cv2.INPAINT_TELEA)

    return base_image1


if __name__ == "__main__":
    # Load the base image (with patch already inserted)
    base_image = cv2.imread('combined_large_image.png')

    # Load the patch image
    patch_image = cv2.imread('patch_image.png')

    # resize patch
    # Remove patch from base image
    base_image_removed_patch = remove_patch(base_image, patch_image)

    base_image_interpolated = interpolate_missing_pixels(base_image_removed_patch)

    # Display the result
    cv2.imshow('Base Image with Patch Removed', base_image_interpolated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
