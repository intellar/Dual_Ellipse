"""!
@file find_ellipses.py
@author Intellar (https://github.com/intellar)
@brief Core ellipse detection logic, including dual conic fitting and gradient processing.

@copyright Copyright (c) 2024

@license See LICENSE for details.

"""

import numpy as np
from scipy.ndimage import convolve1d, label as bwlabel
from PIL import Image
from conic_utils import rebuild_conic, debuild_conic
from typing import Tuple


def dual_conic_fitting(Ix: np.ndarray, Iy: np.ndarray, x: np.ndarray, y: np.ndarray, 
                       normalization: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fits a dual ellipse to the image gradient.

    Args:
        Ix (np.ndarray): Vector containing the horizontal image gradient.
        Iy (np.ndarray): Vector containing the vertical image gradient.
        x (np.ndarray): Corresponding x-coordinates of the image gradient.
        y (np.ndarray): Corresponding y-coordinates of the image gradient.
        normalization (bool): If True, performs data normalization for numerical stability.

    Returns:
        tuple: (dC, std_center, angle_incertitude)
            dC (np.ndarray): The fitted 3x3 dual conic matrix.
            std_center (np.ndarray): Estimated uncertainty on the center.
            angle_incertitude (float): Angle of the center uncertainty covariance ellipse.
    """
    a = Ix
    b = Iy
    c = -(Ix * x + Iy * y)
    H = np.eye(3)

    if normalization:
        # Data normalization
        # A simpler and effective normalization is to translate to the centroid.
        try:
            mpts = np.array([np.mean(x), np.mean(y)])
            # Translation matrix
            H = np.array([
                [1, 0, mpts[0]],
                [0, 1, mpts[1]],
                [0, 0, 1]
            ])
            L = np.vstack([a, b, c])
            # Transform line parameters for the normalized system.
            L_norm = H.T @ L
            a, b, c = L_norm[0, :], L_norm[1, :], L_norm[2, :]
        except np.linalg.LinAlgError:
            # Normalization failed, proceed without it
            pass

    # Design matrix for least-squares Ax=b with constraint F=1
    D = np.vstack([a**2, a*b, b**2, a*c, b*c, c**2]).T
    
    # To solve the least-squares problem X * sol = y_vec, we explicitly form the
    # normal equations: (X.T @ X) * sol = (X.T @ y_vec).
    # Here, X corresponds to D[:, :5] and y_vec to -D[:, 5].
    X = D[:, :5] 
    c_squared = D[:, 5] # This is c^2 from the design matrix
    
    AA = X.T @ X
    BB = -X.T @ c_squared
    
    try:
        # Solve the normal equations: AA * sol = BB
        # Use a robust SVD-based pseudo-inverse for numerical stability.
        U, S_vals, V_t = np.linalg.svd(AA)
        S_inv = np.zeros_like(S_vals)
        # Use a tolerance to avoid division by zero for small singular values
        S_inv[S_vals > 1e-12] = 1 / S_vals[S_vals > 1e-12]
        
        # Reconstruct pseudo-inverse: inv(AA) = V * diag(S_inv) * U.T
        inv_AA = V_t.T @ np.diag(S_inv) @ U.T
        sol = np.append(inv_AA @ BB, 1) # Append F=1
    except np.linalg.LinAlgError:
        return np.full((3, 3), -1), np.array([-1, -1]), -1

    # Denormalization
    dC_norm = rebuild_conic(sol)
    if normalization:
        dC = H @ dC_norm @ H.T
    else:
        dC = dC_norm

    # Error estimation

    s_vec = sol[:5]
    BTB = np.sum(c_squared**2) # This is sum((c^2)^2) = sum(c^4)
    
    # Residual sum of squares
    R = (s_vec.T @ AA @ s_vec - 2 * s_vec.T @ BB + BTB) / max(1, len(a) - 5)
    cvar2_constant_variance = R * inv_AA
    
    cov_DE = cvar2_constant_variance[3:5, 3:5]
    
    try:
        U, S, V_t = np.linalg.svd(cov_DE)
        V = V_t.T
    except np.linalg.LinAlgError:
        return dC, np.array([-1, -1]), -1

    # Center of dual conic is [-D/2F, -E/2F]. With F=1, it's [-D/2, -E/2].
    # The parameters in `sol` are [A, B, C, D, E].
    # Variance of center is Var([-D/2, -E/2]) = (1/4) * Var([D, E])
    std_center = np.sqrt(S) / 2
    angle_incertitude = np.arctan2(V[1, 0], V[0, 0])

    return dC, std_center, angle_incertitude


def find_ellipses(img: np.ndarray) -> np.ndarray:
    """
    Finds ellipses in a grayscale image.

    Args:
        img (np.ndarray): Input image (can be RGB or grayscale).

    Returns:
        np.ndarray: An array of found ellipses, where each row is a 6-element
                    parameter vector [A, B, C, D, E, F].
    """
    # --- Constants for the algorithm ---
    MIN_SEGMENT_PIXELS = 20
    GRADIENT_THRESHOLD_FACTOR = 3.0
    CENTER_STD_DEV_THRESHOLD = 0.15

    if img.ndim == 3 and img.shape[2] > 1:
        # Convert to grayscale using Pillow's functionality
        img = np.array(Image.fromarray(img).convert('L'))

    # Derivative kernels (approximating Gaussian derivatives)
    d = np.array([0.2707, 0.6065, 0, -0.6065, -0.2707])
    g = np.array([0.1353, 0.6065, 1.0, 0.6065, 0.1353])

    # Use separable 1D convolutions for efficiency, which is equivalent to 2D convolution with the outer product.
    # This is generally faster for image processing tasks.
    # For dx: smooth vertically (axis 0), differentiate horizontally (axis 1)
    dx = convolve1d(convolve1d(img.astype(float), g, axis=0, mode='reflect'), d, axis=1, mode='reflect')
    # For dy: differentiate vertically (axis 0), smooth horizontally (axis 1)
    dy = convolve1d(convolve1d(img.astype(float), d, axis=0, mode='reflect'), g, axis=1, mode='reflect')

    grad = np.sqrt(dx**2 + dy**2)
    
    # Remove boundary gradients
    grad[:, :4] = 0
    grad[:4, :] = 0
    grad[:, -4:] = 0
    grad[-4:, :] = 0

    s_img = img.shape
    # Use 'xy' (Cartesian) indexing, which is the default for meshgrid.
    # This matches the convention of the gradient operators:
    # X will contain column indices (horizontal coordinate, matching dx).
    # Y will contain row indices (vertical coordinate, matching dy).
    X, Y = np.meshgrid(np.arange(s_img[1]), np.arange(s_img[0]))

    # Use only strong gradients for segmentation
    mask = grad > GRADIENT_THRESHOLD_FACTOR * np.mean(grad)
    # bwlabel from scipy.ndimage is equivalent to skimage's with connectivity=2 by default
    # It returns the labeled array and the number of labels found.
    labeled_mask, n_labels = bwlabel(mask)

    # Pre-calculate segment sizes to quickly filter out small ones
    segment_sizes = np.bincount(labeled_mask.ravel())

    ellipses = []
    for i in range(1, n_labels + 1):
        if segment_sizes[i] < MIN_SEGMENT_PIXELS:
            continue

        ind_loc = np.where(labeled_mask == i)
        dC, std_center, angle_incertitude = dual_conic_fitting(
            dx[ind_loc], dy[ind_loc], X[ind_loc], Y[ind_loc]
        )
        
        # Filter out fits with high uncertainty in the center position.
        # A low standard deviation indicates a more reliable ellipse fit.
        if not (std_center[0] == -1 or std_center[0] > CENTER_STD_DEV_THRESHOLD):
            try:
                C = np.linalg.inv(dC)
                C = C / C[2, 2]  # Normalize
                ellipse_params = debuild_conic(C)
                ellipses.append(ellipse_params)
            except np.linalg.LinAlgError:
                pass # Inversion failed, not a valid ellipse
            
    return np.array(ellipses)
