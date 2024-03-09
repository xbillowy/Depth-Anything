import cv2
import numpy as np


def rotate_image(image, angle):
    # Deal with nasty shapes
    sh, h, w = image.shape, image.shape[0], image.shape[1]
    
    # Compute the center of the image
    center = (w / 2, h / 2)
    
    # Perform the rotation using `cv2.getRotationMatrix2D`
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the actual rotation using `cv2.warpAffine`
    rotated = cv2.warpAffine(image, M, (w, h))

    # Reshape the image to the original shape
    rotated = rotated.reshape(sh)

    return rotated
