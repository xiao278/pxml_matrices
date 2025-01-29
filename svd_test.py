import colorsys
import numpy as np
from numpy import linalg as LA
from PIL import Image

IMAGE_PATH = 'shanghai.png'
TARGET_RANK = 150

np.set_printoptions(suppress=True)

def channels_first(image):
    return image.swapaxes(2, 1).swapaxes(1, 0)

def channels_last(image):
    return image.swapaxes(0, 1).swapaxes(1, 2)

def rgb_to_hsv(image_arr):
    x, y, _ = image_arr.shape
    return np.stack([colorsys.rgb_to_hsv(*color) for color in image_arr.reshape(-1, 3)]).reshape(x, y, 3)
    
def hsv_to_rgb(image_arr):
    x, y, _ = image_arr.shape
    return np.stack([colorsys.hsv_to_rgb(*color) for color in image_arr.reshape(-1, 3)]).reshape(x, y, 3)

def svd(A, k):
    """Computes the rank-k reduced SVD of the matrix A."""
    eigenvalues, eigenvectors = LA.eig(A.T @ A);
    
    # eigenvectors[i] gets the ith row, not the ith eigenvector.
    # to fix this, we take the transpose.
    eigenvectors = eigenvectors.T
    
    # sort the eigenvectors by their singular values descending
    eigenvalues, eigenvectors = zip(*sorted(zip(abs(eigenvalues), eigenvectors), key=lambda x : x[0], reverse=True))

    # sigma: square matrix of singular values
    S = np.diag(np.sqrt(eigenvalues[:k]))
    
    # V: matrix of first k right singular vectors
    V = np.stack(eigenvectors[:k], axis=1)
    
    # U: matrix of first k left singular vectors
    U = np.stack([A @ eigenvectors[i] / S[i][i] for i in range(k)], axis=1)

    return U, S, V

def svd_reduce(A, k):
    U, S, V = svd(A, k)
    return U @ S @ V.T

def main():
    A = np.array([[3, 2, 2], [2, 3, -2]])
    U, S, V = svd(A, 2);
    
    image = Image.open(IMAGE_PATH)
    # image.show()

    image_arr = np.array(image) / 255;

    image_arr = rgb_to_hsv(image_arr)
    
    recon_arr = channels_last(np.stack([svd_reduce(channel, TARGET_RANK) for channel in channels_first(image_arr)]))

    recon_arr = hsv_to_rgb(recon_arr)
    recon_arr = recon_arr * 255
    recon_arr = recon_arr.clip(0, 255).astype(np.uint8)
    
    recon = Image.fromarray(recon_arr)
    recon.show()
    
if __name__ == '__main__':
    main()