import matplotlib.pyplot as plt
import helpers as helpers
import scipy.optimize as opt
import numpy as np

def lstsq(x, a, b):
    return (1/2) * np.linalg.norm(np.dot(a, x) - b)**2

def grad(x, a, b):
    return np.dot(a.T, np.dot(a, x) - b)


def blur_datasource(image_path, sigma, ker_len):
    # Open the image as a matrix and show it
    picture = plt.imread(image_path) 
    picture = picture[:, : , 0]
    n, m = picture.shape

    # Get the kernel
    kernel = helpers.gaussian_kernel(ker_len, sigma)

    # Appply Fourier
    k = np.real(helpers.psf_fft(kernel, ker_len, (n, m)))

    # Blur and show
    blurred = helpers.A(picture, k)
    plt.imshow(blurred, cmap='gray') 
    plt.show()

    # Reshape
    picture_vec = picture.reshape(n * m)
    k_vec = k.reshape(n * m)
    blurred_vec = blurred.reshape(n * m)

    f = lambda x: lstsq(x, k_vec, blurred_vec)
    grad_f = lambda x: grad(x, k_vec, blurred_vec)

    x0 = np.zeros(n * m)

    minimum = opt.minimize(f, x0, jac=grad_f, method='CG')

    plt.imshow(minimum.x.reshape((n, m)), cmap='gray') 
    plt.show()

    return blurred


def main():
    ds1 = './datasource/one.png'

    # TODO: add gaussian rumor between ]0; 0,05]
    b1_ds1 = blur_datasource(ds1, 0.5, 5)
    # b2_ds1 = blur_datasource(ds1, 1, 7)
    # b3_ds1 = blur_datasource(ds1, 1.3, 9)
    
    # Add here calls to phase2


if __name__ == "__main__":
    main()
