import cv2
import numpy as np

from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform
from myHoughLines import myHoughLines


# Read image
img0 = cv2.imread('../data/img10.jpg', cv2.IMREAD_GRAYSCALE)

if img0 is None:
    print("Could not load image")
    exit()

# Convert to float
img0 = np.float32(img0) / 255.0

print("Image shape:", img0.shape)


# Parameters
sigma = 2
threshold = 0.03

rhoRes = 1.0
thetaRes = 0.01

nLines = 10


# Edge detection
img_edge = myEdgeFilter(img0, sigma)

# Threshold edge image
img_threshold = np.float32(img_edge > threshold)


# Hough Transform

img_hough, rhoScale, thetaScale = myHoughTransform(
    img_threshold,
    rhoRes,
    thetaRes
)


# Find strongest Hough lines
rhos, thetas = myHoughLines(
    img_hough,
    nLines
)


# Print results
print("Rho indices:")
print(rhos)

print("Theta indices:")
print(thetas)


# Save Hough accumulator
cv2.imwrite(
    'results/houghlines10.jpg',
    np.uint8(255 * img_hough / img_hough.max())
)

print("Hough transform complete")
print("Saved result as results/houghlines10.jpg")