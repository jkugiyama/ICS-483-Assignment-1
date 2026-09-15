import numpy as np

def myImageFilter(img0, h):

    # Get image dimensions
    img_height, img_width = img0.shape

    # Get filter dimensions
    h_height, h_width = h.shape

    # Calculate padding
    pad_height = h_height // 2
    pad_width = h_width // 2

    # Pad image using nearest pixel values
    img_pad = np.pad(
        img0,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode='edge'
    )

    # Create output image
    img1 = np.zeros_like(img0, dtype=np.float32)

    # Flip filter for convolution
    h_flip = np.flip(h)

    # Apply filter
    for i in range(h_height):
        for j in range(h_width):
            img1 += h_flip[i, j] * img_pad[
                i:i + img_height,
                j:j + img_width
            ]

    return img1