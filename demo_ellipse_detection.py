"""!
@file demo_ellipse_detection.py
@author Intellar (https://github.com/intellar)
@brief Main script to run the dual ellipse fitting demonstration.

@copyright Copyright (c) 2024

@license See LICENSE for details.

"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from find_ellipses import find_ellipses
from draw_conic import draw_conic

def demo_ellipse_detection():
    """
    Loads an image, detects ellipses, and displays them.
    This function replicates the behavior of DemoEllipseDetection.m.
    """
    # Load the image
    try:
        # Using Pillow to open the image and convert to a NumPy array
        img = np.array(Image.open('Ford.bmp'))
    except FileNotFoundError:
        print("Error: 'Ford.bmp' not found.")
        print("Please make sure the image file is in the same directory as the script.")
        return

    # Find ellipses in the image
    print("Detecting ellipses...")
    ellipses = find_ellipses(img)
    print(f"Found {len(ellipses)} potential ellipses.")

    # Create a figure and display the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    # Draw each detected ellipse
    if ellipses.size > 0:
        for ellipse_params in ellipses:
            draw_conic(ellipse_params, color='g', linewidth=2, ax=ax)

    plt.show()

if __name__ == '__main__':
    demo_ellipse_detection()