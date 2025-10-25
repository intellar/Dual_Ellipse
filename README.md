# Dual Ellipse Fitting in Python

This project is a Python implementation of the robust ellipse fitting algorithm described in the 2008 paper **"Accurate and robust ellipse fitting in real-time"** by Jean-Nicolas Ouellet and Patrick Hébert. It was ported from the original MATLAB source code provided by the author.

The algorithm fits ellipses to image gradients by representing them as a set of tangent lines and fitting a *dual conic*, which provides superior robustness and accuracy compared to traditional point-based (primal) fitting methods.



## Features

-   **Dual Conic Fitting:** Utilizes the dual representation of conics for robust fitting to tangent lines.
-   **Robust Error Estimation:** Implements the paper's method for estimating the uncertainty of the ellipse center, allowing for effective filtering of unreliable fits.
-   **Numerically Stable:** Employs data normalization to ensure the stability of the least-squares fitting process.
-   **Clean and Modular Code:** The implementation is structured into clear, purpose-driven Python modules.

## Requirements

-   Python 3.x
-   NumPy
-   SciPy
-   Pillow
-   Matplotlib

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Dual_Ellipse.git
    cd Dual_Ellipse
    ```

2.  Install the required packages using pip. A `requirements.txt` file is provided for convenience:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the demonstration on the sample image (`Ford.bmp`), simply execute the `demo_ellipse_detection.py` script:

```bash
python demo_ellipse_detection.py
```

This will load the image, detect all ellipses, and display the results using Matplotlib.

## Project Structure

*   `demo_ellipse_detection.py`: The main script to run the demonstration.
*   `find_ellipses.py`: Contains the core ellipse detection logic, including gradient calculation, segmentation, and the fitting loop.
*   `draw_conic.py`: Utility functions to draw the detected ellipses on a plot.
*   `conic_utils.py`: Helper functions for converting between conic matrix and parameter vector representations.
*   `Ford.bmp`: The sample image used in the demo.
*   `requirements.txt`: A list of Python package dependencies.

## Reference

Ouellet, J. N., & Hebert, P. (2008). Accurate and robust ellipse fitting in real-time. *2008 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops*. DOI: 10.1109/CVPRW.2008.4563057