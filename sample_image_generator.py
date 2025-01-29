import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
from scipy.optimize import curve_fit


def plot_heatmap(matrix, title='Heatmap', cmap='viridis', annot=False):
    """
    Plots a heatmap of the given matrix using only Matplotlib.
    
    Parameters:
        matrix (ndarray): 2D NumPy array representing the matrix.
        title (str): Title of the heatmap.
        cmap (str): Colormap for the heatmap.
        annot (bool): If True, display values in the heatmap.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    
    if annot:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center', color='black')
    
    plt.show()


def plot_matrix(matrix1, matrix2, title="Heatmap"):
    fig = plt.figure(figsize=(10,6))
    gs = gridspec.GridSpec(1,2)

    ax1 = fig.add_subplot(gs[0,0])
    im = ax1.imshow(matrix1, cmap=None, interpolation='nearest')
    fig.colorbar(im, ax=ax1)
    ax1.set_title(title, fontsize=16)
    ax1.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    ax2 = fig.add_subplot(gs[0,1])
    im = ax2.imshow(matrix2, cmap=None, interpolation='nearest')
    fig.colorbar(im, ax=ax2)
    ax2.set_title(title, fontsize=16)
    ax2.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(title + ".png")
    plt.show()



def generate(xdim, ydim):

    m = np.zeros((xdim, ydim))
    m_shifted = np.zeros((xdim, ydim))

    size = np.random.uniform(0.1, 0.2)
    xpos = np.random.uniform(-0.2, 0.2)
    ypos = np.random.uniform(-0.2, 0.2)
    noise_level = 1

    xshift = np.random.uniform(-0.1, 0.1)
    yshift = np.random.uniform(-0.1, 0.1)
    
    for x in range(xdim):
        xx = x / xdim - 1/2 
        for y in range(ydim):
            yy = y / ydim - 1/2
            noise_1 = np.random.uniform(-noise_level, noise_level)
            noise_2 = np.random.uniform(-noise_level, noise_level)
            m[x,y] = 4*scipy.stats.norm.pdf(xx, xpos, size) * scipy.stats.norm.pdf(yy, ypos, size) + noise_1
            m_shifted[x,y] = 4*scipy.stats.norm.pdf(xx, xpos + xshift, size) * scipy.stats.norm.pdf(yy, ypos + yshift, size) + noise_2
    
    return (m, m_shifted)





def gaussian_fit(matrix):

    def gaussian_2d(X, A, x0, y0, sigma, offset):
        x, y = X
        return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + offset


    x = np.linspace(-0.5, 0.5, 25)
    y = np.linspace(-0.5, 0.5, 25)

    X, Y = np.meshgrid(x,y)

    x_data = X.ravel()
    y_data = Y.ravel()
    z_data = matrix.ravel()

    p0 = [4, 0, 0, 0.4, 0]


    popt, _ = curve_fit(gaussian_2d, (x_data, y_data), z_data, p0=p0)

    #A_fit, x0_fit, y0_fit, sigma_fit, offset_fit = popt
    print(popt)

    Z_fit = gaussian_2d((X, Y), *popt)
    
    return Z_fit



if __name__ == "__main__":
    #data = np.random.rand(10, 10)  # Generate a 10x10 random matrix
    (a,b) = generate(25,25)

    c = gaussian_fit(a)

    #np.save("Matrix_1", a)
    #np.save("Matrix_1_shifted", b)

    plot_matrix(a, c)