"""!
@file draw_conic.py
@author Intellar (https://github.com/intellar)
@brief Utility functions to draw detected ellipses and convert conic parameters.

@copyright Copyright (c) 2024

@license See LICENSE for details.

"""

import numpy as np
import matplotlib.pyplot as plt
from conic_utils import debuild_conic
from typing import Tuple, Union
 

def convert_conic_to_parametric(par: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Converts conic parameters from the implicit form
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0 to the parametric form
    (center_x, center_y, axis_a, axis_b, angle).

    Args:
        par (np.ndarray): A 6-element array [A, B, C, D, E, F] or a 3x3 conic matrix.

    Returns:
        tuple: (center_x, center_y, axis_a, axis_b, angle) if a valid ellipse,
               otherwise (-1, -1, 0, 0, 0).
    """
    # Ensure input is a 6-element vector
    if par.shape == (3, 3):
        par = debuild_conic(par)

    A, B, C, D, E, F = par

    # An ellipse is defined by the discriminant being negative
    discriminant = B**2 - 4*A*C
    if discriminant >= 0:
        # Not an ellipse (parabola, hyperbola, or degenerate)
        return -1, -1, 0, 0, 0

    # --- More direct algebraic method for parameter extraction ---

    # 1. Find the center of the ellipse by solving the system
    #    2*A*x + B*y + D = 0
    #    B*x + 2*C*y + E = 0
    # This is equivalent to M * [center_x, center_y]' = V
    M = np.array([[2*A, B], [B, 2*C]])
    V = np.array([-D, -E])
    try:
        center_x, center_y = np.linalg.solve(M, V)
    except np.linalg.LinAlgError:
        # Singular matrix, indicates a degenerate conic
        return -1, -1, 0, 0, 0

    # 2. Translate the conic to the origin to find axes and angle
    # The new constant term F' = A*x^2 + B*x*y + C*y^2 + D*x + E*y + F
    F_prime = A*center_x**2 + B*center_x*center_y + C*center_y**2 + D*center_x + E*center_y + F

    # 3. Find axes lengths and angle from the eigenvalues of the quadratic part
    # The equation at the origin is A*x^2 + B*x*y + C*y^2 = -F'
    # The eigenvalues of the quadratic matrix give the scaled inverse-squared axes
    quad_matrix = np.array([[A, B/2], [B/2, C]])
    eigenvalues, eigenvectors = np.linalg.eigh(quad_matrix)
    
    if F_prime == 0 or np.any(eigenvalues <= 0):
        return -1, -1, 0, 0, 0 # Degenerate case

    axis_a = np.sqrt(-F_prime / eigenvalues[1])
    axis_b = np.sqrt(-F_prime / eigenvalues[0])
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])

    return center_x, center_y, axis_a, axis_b, angle


def draw_conic(conic_params: np.ndarray, 
               color: str = 'r', linewidth: int = 1, 
               ax: Union[plt.Axes, None] = None) -> None:
    """
    Draws an ellipse on a matplotlib axes.

    Args:
        conic_params (np.ndarray): Conic parameters. Can be a 6-element vector [A,B,C,D,E,F],
                                   a 3x3 matrix, or a 5-element parametric vector
                                   [center_x, center_y, axis_a, axis_b, angle].
        color (str): Color for the ellipse outline.
        linewidth (int): Line width for the ellipse outline.
        ax (matplotlib.axes.Axes, optional): The axes to draw on. If None, uses the current axes.
    """
    if ax is None:
        ax = plt.gca()

    # The function can handle a 6-element vector or a 3x3 matrix directly.
    # A 5-element vector is assumed to be already in parametric form.
    if conic_params.shape == (5,):
        parametric_params = conic_params
    else:
        parametric_params = convert_conic_to_parametric(conic_params)

    # convert_conic_to_parametric returns center_x = -1 for non-ellipses
    if parametric_params[0] == -1:
        return
    
    center_x, center_y, axis_a, axis_b, angle = parametric_params
    # Generate points for the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = axis_a * np.cos(theta)
    ellipse_y = axis_b * np.sin(theta)

    # Rotate and translate the points
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = R @ np.vstack([ellipse_x, ellipse_y])
    x = points[0, :] + center_x
    y = points[1, :] + center_y

    ax.plot(x, y, color=color, linewidth=linewidth)