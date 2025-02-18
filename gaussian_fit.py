import numpy as np
import scipy
from scipy.optimize import curve_fit


def find_peak(matrix):
    """
    Function that fits a 2d Gaussian function to the input matrix and returns the peak (x,y) of the
    function.
    """
    
    # Function for fitting
    def gaussian_2d(X, A, x0, y0, sigma, offset):
        x, y = X
        return A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)) + offset

    # Create arrays for fitting
    x = np.array(range(matrix.shape[0]))
    y = np.array(range(matrix.shape[1]))

    X, Y = np.meshgrid(x,y)
    x_data = X.ravel()
    y_data = Y.ravel()
    z_data = matrix.ravel()

    # Initial guess
    mode = scipy.stats.mode(matrix.flatten())[0][0]
    top = np.mean(np.sort(matrix.flatten())[-5:])

    A_guess = top - mode
    offset_guess = mode

    # Estimate sigma
    matrix_row_sum = matrix.sum(axis=0)
    matrix_row_sum = matrix_row_sum - np.mean(matrix_row_sum)
    matrix_row_sum[matrix_row_sum < 0] = 0

    mean = np.sum(x * matrix_row_sum) / np.sum(matrix_row_sum)
    variance = np.sum(matrix_row_sum * (x - mean) ** 2) / np.sum(matrix_row_sum)
    sigma_guess = np.sqrt(variance)

    #p0 = [A, x0, y0, sigma, offset]
    p0 = [A_guess, matrix.shape[0]//2, matrix.shape[1]//2, sigma_guess, offset_guess]
    
    # Fit curve to data
    popt, _ = curve_fit(gaussian_2d, (x_data, y_data), z_data, p0=p0)

    # Return peak of gaussian
    return (int(round(popt[1])), int(round(popt[2])))



    