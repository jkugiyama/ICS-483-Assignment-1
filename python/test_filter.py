import cv2
import numpy as np
from myImageFilter import myImageFilter

# Read the image
img0 = cv2.imread('../data/img01.jpg', cv2.IMREAD_GRAYSCALE)

# Convert to float
img0 = np.float32(img0) / 255.0

# Create a simple 3x3 averaging filter
h = np.ones((3, 3), dtype=np.float32) / 9

print("Image shape:", img0.shape)
print("Filter shape:", h.shape)

# Apply the filter
img1 = myImageFilter(img0, h)

# Save the result
cv2.imwrite('filtered.jpg', np.uint8(img1 * 255))

print("Filtering complete!")
print("Saved result as filtered.jpg")