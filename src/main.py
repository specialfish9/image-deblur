import matplotlib.pyplot as plt
import helpers as helpers
import scipy.optimize as opt
import numpy as np
from skimage import data, metrics

def lstsq(x, a, b):
    return (1/2) * np.linalg.norm(helpers.A(x, a) - b) ** 2

def grad(x, a, b):
    return helpers.AT(helpers.A(x, a), a) - helpers.AT(b, a)

def naive_deblur(blurred, n, m, k):
    # Function to minimize
    f = lambda x: lstsq(x.reshape(n, m), k, blurred)
    # Gradient of f
    grad_f = lambda x: grad(x.reshape(n, m), k, blurred).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    max_iter = 50
    
    minimum = opt.minimize(f, x0, jac=grad_f, options={'maxiter':max_iter}, method='CG')
    return minimum.x.reshape((n, m))

def blur_picture(picture, k):
    # Generate noise
    s = 0.004
    noise = np.random.normal(size=picture.shape) * s
    
    # Blur and add noise
    return helpers.A(picture, k) + noise


def fReg(x, a, b, lamb=1.5):
  return (1/2) * (np.linalg.norm(helpers.A(x, a)-b)) ** 2 + (1/2) * lamb * np.linalg.norm(x) ** 2

def gradReg(x, a, b, lamb=1.5):
  return helpers.AT(helpers.A(x, a), a) - helpers.AT(b, a) + lamb * x

def regular_deblur(blurred, n, m, k, x=1.5):
    # Function to minimize
    lamb = 1.5
    f = lambda x: fReg(x.reshape(n, m), k, blurred, lamb)
    # Gradient of f
    grad_f = lambda x: gradReg(x.reshape(n, m), k, blurred, lamb).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    max_iter = 50
    
    minimum = opt.minimize(f, x0, jac=grad_f, options={'maxiter':max_iter}, method='CG')
    return minimum.x.reshape((n, m))

def elaborate_datasource(image_path, sigma, ker_len):
    # Open the image as a matrix and show it
    picture = plt.imread(image_path) 
    picture = picture[:, : , 0]
    n, m = picture.shape
    plt.imshow(picture, cmap='gray') 
    plt.show()

    # Create the kernel
    kernel = helpers.gaussian_kernel(ker_len, sigma)
    k = helpers.psf_fft(kernel, ker_len, (n, m))

    # Phase 1: blur
    blurred = blur_picture(picture, k)
    plt.imshow(blurred, cmap='gray') 
    plt.show()

    # Phase 2: naive solution
    deblurred = naive_deblur(blurred, n, m, k)
    plt.imshow(deblurred, cmap='gray') 
    plt.show()

    # Add here code for phase 3
    deblurred3 = regular_deblur(blurred, n, m, k)
    plt.imshow(deblurred3, cmap='gray') 
    plt.show()



def main():
    ds1 = './datasource/six.png'

    # b1_ds1 = elaborate_datasource(ds1, 0.5, 5)
    # b2_ds1 = elaborate_datasource(ds1, 1, 7)
    b3_ds1 = elaborate_datasource(ds1, 1.3, 9)


if __name__ == "__main__":
    main()
