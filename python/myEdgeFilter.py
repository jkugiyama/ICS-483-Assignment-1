import numpy as np
from scipy.signal.windows import gaussian # For signal.gaussian function
from myImageFilter import myImageFilter 

from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    # YOUR CODE HERE
    # Create Gaussian smoothing filter

    # Size of Gaussian filter
    hsize = 2 * int(np.ceil(3 * sigma)) + 1

    # Create 1D Gaussian kernels
    h1 = gaussian(hsize, sigma)

    # Normalize Gaussian kernel
    h1 = h1 / np.sum(h1)

    # Create 2D Gaussian filter
    h = np.outer(h1, h1)

    # Smooth the image

    img_smooth = myImageFilter(img0, h)

    # Create Sobel filters

    # Sobel filter for x direction
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Sobel filter for y direction
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    # Calculate image gradients

    imgx = myImageFilter(img_smooth, sobel_x)
    imgy = myImageFilter(img_smooth, sobel_y)

    # Gradient magnitude
    magnitude = np.sqrt(imgx ** 2 + imgy ** 2)

    # Gradient angle
    angle = np.arctan2(imgy, imgx)

    # Convert angle from radians to degrees
    angle = angle * 180 / np.pi

    # Make angles positive
    angle[angle < 0] += 180

    # Non-maximum suppression

    img1 = np.zeros_like(magnitude)

    # Get image dimensions
    rows, cols = magnitude.shape

    # Ignore the outermost pixels
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            # Current gradient angle
            a = angle[i, j]

            # Determine which direction to compare

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                # 0 degrees
                neighbor1 = magnitude[i, j - 1]
                neighbor2 = magnitude[i, j + 1]

            elif 22.5 <= a < 67.5:
                # 45 degrees
                neighbor1 = magnitude[i - 1, j + 1]
                neighbor2 = magnitude[i + 1, j - 1]

            elif 67.5 <= a < 112.5:
                # 90 degrees
                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]

            else:
                # 135 degrees
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            # Keep pixel only if it is a local maximum

            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                img1[i, j] = magnitude[i, j]

    return img1
