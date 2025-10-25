"""!
@file conic_utils.py
@author Intellar (https://github.com/intellar)
@brief Helper functions for converting between conic matrix and parameter vector representations.

@copyright Copyright (c) 2024

@license See LICENSE for details.

"""

import numpy as np

def rebuild_conic(par: np.ndarray) -> np.ndarray:
    """
    Rebuilds a 3x3 conic matrix from a 6-element parameter vector.
    The parameter vector is [A, B, C, D, E, F] for the conic equation
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
    The dual conic matrix is for the line equation a*u + b*v + c*w = 0.
    The matrix Q is for A*u^2 + B*uv + C*v^2 + D*uw + E*vw + F*w^2 = 0.

    Args:
        par (np.ndarray): A 6-element array [A, B, C, D, E, F].

    Returns:
        np.ndarray: A 3x3 symmetric conic matrix.
    """
    A, B, C, D, E, F = par
    return np.array([
        [A, B / 2, D / 2],
        [B / 2, C, E / 2],
        [D / 2, E / 2, F]
    ])


def debuild_conic(Q: np.ndarray) -> np.ndarray:
    """
    Extracts the 6-element parameter vector from a 3x3 conic matrix.

    Args:
        Q (np.ndarray): A 3x3 symmetric conic matrix.

    Returns:
        np.ndarray: A 6-element array [A, B, C, D, E, F].
    """
    return np.array([
        Q[0, 0],
        Q[0, 1] * 2,
        Q[1, 1],
        Q[0, 2] * 2,
        Q[1, 2] * 2,
        Q[2, 2]
    ])