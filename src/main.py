import matplotlib.pyplot as plt
import helpers as helpers
import scipy.optimize as opt
import numpy as np

def lstsq(x, a, b):
    return (1/2) * np.linalg.norm(a @ x - b)**2


def blur_datasource(image_path, sigma, ker_len):
    # Open the image as a matrix and show it
    picture = plt.imread(image_path) 
    picture = picture[:, : , 0]

    # Get the kernel
    kernel = helpers.gaussian_kernel(ker_len, sigma)

    # Appply Fourier
    k = helpers.psf_fft(kernel, ker_len, (512,512))

    # Blur and show
    blurred = helpers.A(picture, k)
    plt.imshow(blurred, cmap='gray') 
    plt.show()

    # Reshape
    n, m = picture.shape
    picture_vec = picture.reshape(n * m)
    k_vec = k.reshape(n * m)
    blurred_vec = blurred.reshape(n * m)

    def f(x):
        if len(x.shape) != 2:
            x = x.reshape((n, m))
        return lstsq(x, k, blurred)

    x0 = np.zeros((n, m))

    minimum = opt.minimize(f, x0, method='CG')

    plt.imshow(minimum.x.reshape((n, m)), cmap='gray') 
    plt.show()

    return blurred


def main():
    ds1 = './datasource/e1.png'

    # TODO: add gaussian rumor between ]0; 0,05]
    b1_ds1 = blur_datasource(ds1, 0.5, 5)
    b2_ds1 = blur_datasource(ds1, 1, 7)
    b3_ds1 = blur_datasource(ds1, 1.3, 9)
    
    # Add here calls to phase2


if __name__ == "__main__":
    main()
