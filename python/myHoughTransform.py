import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    # Get image dimensions
    M, N = Im.shape

    # Create theta and rho scales

    # Theta ranges from 0 to 2*pi
    thetaScale = np.arange(0, 2 * np.pi, thetaRes)

    # Maximum possible rho
    # This is the diagonal length of the image
    rhoMax = np.sqrt(M**2 + N**2)

    # Rho ranges from 0 to maximum possible distance
    rhoScale = np.arange(0, rhoMax + rhoRes, rhoRes)

    # Create Hough accumulator

    img_hough = np.zeros(
        (len(rhoScale), len(thetaScale)),
        dtype=np.float32
    )

    y, x = np.nonzero(Im)

    # Vote in the Hough accumulator

    for j in range(len(thetaScale)):

        theta = thetaScale[j]

        # Calculate rho for every edge pixel
        rho = x * np.cos(theta) + y * np.sin(theta)

        # Only keep positive rho values
        valid = rho >= 0

        rho_valid = rho[valid]

        # Convert rho values to accumulator indices
        rho_indices = np.round(rho_valid / rhoRes).astype(int)

        # Remove indices outside the accumulator
        valid_indices = rho_indices < len(rhoScale)

        rho_indices = rho_indices[valid_indices]

        # Add votes
        np.add.at(
            img_hough[:, j],
            rho_indices,
            1
        )

    return img_hough, rhoScale, thetaScale