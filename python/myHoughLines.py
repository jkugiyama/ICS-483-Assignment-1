import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    # YOUR CODE HERE

    # Create a copy of the Hough accumulator

    hough = H.copy()

    # Non-maximum suppression

    # Size of neighborhood used to find local maxima
    neighborhood_size = 21

    kernel = np.ones(
        (neighborhood_size, neighborhood_size),
        np.uint8
    )

    # Find the maximum value in each neighborhood
    local_max = cv2.dilate(hough, kernel)

    # Keep only pixels that are equal to the local maximum
    peaks = np.where(hough == local_max, hough, 0)

    # Find the strongest peaks

    # Flatten the accumulator
    flat_peaks = peaks.ravel()

    # Sort from largest to smallest
    indices = np.argsort(flat_peaks)[::-1]

    # Keep the requested number of lines
    indices = indices[:nLines]

    # Convert flattened indices back to row/column coordinates
    rhos, thetas = np.unravel_index(
        indices,
        peaks.shape
    )

    # Convert to column vectors
    rhos = rhos.reshape(-1, 1)
    thetas = thetas.reshape(-1, 1)

    return rhos, thetas