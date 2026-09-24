import cv2
import numpy as np
from myHoughTransform import myHoughTransform

# Read the image
img0 = cv2.imread('data/img10.jpg', cv2.IMREAD_GRAYSCALE)

# Convert to float
img0 = np.float32(img0) / 255.0

# Create a simple 3x3 averaging filter
h = np.ones((3, 3), dtype=np.float32) / 9

print("Image shape:", img0.shape)
print("Filter shape:", h.shape)

# Apply the filter
img1, rhoScale, thetaScale = myHoughTransform(img0, rhoRes=1.0, thetaRes=0.01)

# Save the result
cv2.imwrite('ec/hough10.jpg', np.uint8(img1 * 255))

print("Filtering complete!")
print("Saved edges as ec/hough10.jpg")