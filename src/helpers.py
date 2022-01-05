import numpy as np
from numpy import fft


# Create a Gaussian kernel of size kernlen and standard deviation sigma
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)
    # Unidimensional Gaussian kernel
    kern1d = np.exp(- 0.5 * (x ** 2 / sigma))
    # Bidimensional Gaussian kernel
    kern2d = np.outer(kern1d, kern1d)
    # Normalization
    return kern2d / kern2d.sum()


# Compute the FFT of the kernel 'K' of size 'd' padding with the zeros necessary
# to match the size of 'shape'
def psf_fft(K, d, shape):
    # Zero padding
    K_p = np.zeros(shape)
    K_p[:d, :d] = K

    # Shift
    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

    # Compute FFT
    K_otf = fft.fft2(K_pr)
    return K_otf


# Multiplication by A
def A(x, K):
    x = fft.fft2(x)
    return np.real(fft.ifft2(K * x))


# Multiplication by A transpose
def AT(x, K):
    x = fft.fft2(x)
    return np.real(fft.ifft2(np.conj(K) * x))
