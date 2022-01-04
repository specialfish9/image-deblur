import matplotlib.pyplot as plt
import helpers as helpers
import scipy.optimize as opt
import numpy as np


def lstsq(x, a, b):
    """
    least-squares function implementation
    """
    return (1 / 2) * np.linalg.norm(helpers.A(x, a) - b) ** 2


def grad(x, a, b):
    """
    least-squares gradient implementation
    """
    return helpers.AT(helpers.A(x, a), a) - helpers.AT(b, a)


def lstsq_reg(x, a, b, lamb=1.5):
    """
    lstsq with Tikhonov regularization
    """
    return lstsq(x, a, b) + (1 / 2) * lamb * np.linalg.norm(x) ** 2


def grad_reg(x, a, b, lamb=1.5):
    """
    least-squares gradient with Tikhonov regularization
    """

    return grad(x, a, b) + lamb * x


def total_variation(u, n, m, eps=0.01):
    gradient = 0
    for i in range(n):
        for j in range(m):
            gradient += np.sqrt(np.linalg.norm(np.gradient(u, axis=0)[0][i] + np.gradient(u, axis=1)[1][j]) ** 2 + eps ** 2)
    return gradient


def lstsq_totvar(x, a, b, n, m, eps=0.01, lamb=1.5):
    return lstsq(x, a, b) + lamb * total_variation(x, n, m, eps)


def div(f):
    return np.gradient(f, axis=0)[0] + np.gradient(f, axis=1)[1]  # TODO


def grad_totvar(u, eps=0.01):
    return -div(np.gradient(u) / np.sqrt(np.linalg.norm(np.gradient(u)) ** 2 + eps ** 2)) @ u


def grad_lstsq_totvar(x, a, b, eps=0.01, lamb=1.5):
    return grad(x, a, b) + lamb * grad_totvar(x, eps)


def blur_picture(picture, k, s=0.004):
    """
    Blur a picture using gaussian-blur and gaussian noise
    @param picture: Original picture
    @param k: Kernel matrix
    @param s: Standard deviation of gaussian noise
    @return: Blurred picture
    """

    # Generate noise
    noise = np.random.normal(size=picture.shape) * s

    # Blur and add noise
    return helpers.A(picture, k) + noise


def naive_deblur(blurred, n, m, k, max_iter=50):
    """
    Deblur using naive method: find the absolute minimum of lstsq(x, k, blurred) using conjugate gradient
    @param blurred: Blurred image
    @param n: width
    @param m: height
    @param k: Gaussian blur matrix
    @param max_iter: Maximum number of iterations
    @return: de-blurred image
    """

    # Function to minimize
    f = lambda x: lstsq(x.reshape(n, m), k, blurred)
    # Gradient of f
    grad_f = lambda x: grad(x.reshape(n, m), k, blurred).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    minimum = opt.minimize(f, x0, jac=grad_f, options={'maxiter': max_iter}, method='CG')
    return minimum.x.reshape((n, m))


def regular_deblur_cg(blurred, n, m, k, lamb=1.5, max_iter=50):
    """
    Deblur image finding the absolute minimum of lstsq  with Tikhonov regularization
    @param blurred: Image to de-blur
    @param n: Width
    @param m: Height
    @param k: Gaussian-kernel
    @param lamb: regularization term
    @param max_iter: Maximum number of iterations
    @return: De-blurred image
    """

    # Function to minimize
    f = lambda x: lstsq_reg(x.reshape(n, m), k, blurred, lamb)
    # Gradient of f
    grad_f = lambda x: grad_reg(x.reshape(n, m), k, blurred, lamb).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    minimum = opt.minimize(f, x0, jac=grad_f, options={'maxiter': max_iter}, method='CG')
    return minimum.x.reshape((n, m))


def next_step(x, f, grad_x, n, m):
    """
      Find next step for gradient descent method
      @param x: Current x_k (immutable)
      @param f: Target function
      @param grad_x: Gradient function evaluated on x_k
      @param n: width
      @param m: height
      @return: Next step
    """
    alpha = 1.1
    rho = 0.5
    c1 = 0.25
    p = -grad_x
    j = 0
    jmax = 10
    while (f(x.reshape(n, m) + alpha * p) > f(x.reshape(n, m)) + c1 * alpha * (
            grad_x.reshape(n * m).T @ p.reshape(n * m))) and j < jmax:
        alpha = rho * alpha
        j += 1

    if j > jmax:
        return -1
    else:
        return alpha


def gradient_descend(f, n, m, grad_f, x0, max_iter=50, absolute_stop=1.e-5):
    k = 0

    while np.linalg.norm(grad_f(x0)) > absolute_stop and k < max_iter:
        k += 1
        gradient = grad_f(x0)
        alpha = next_step(x0, f, gradient, n, m)

        if alpha == -1:
            print("No convergence")
            return None

        x0 = x0 - alpha * gradient.reshape(n * m)

    return x0.reshape(n, m)


def regular_deblur_gradient(blurred, n, m, ker, max_iter=50, absolute_stop=1.e-5):
    """
      Deblur an image using Gradient Descend method
      @param blurred: Blurred image
      @param n: Width
      @param m: Height
      @param ker: Gaussian kernel matrix
      @param max_iter: Maximum number of iterations
      @param absolute_stop: Absolute stop value
      @return: De-blurred image
    """

    # initialize first values
    x0 = np.zeros(n * m)
    x0[0] = 1

    f = lambda x: lstsq_reg(x.reshape(n, m), ker, blurred)
    grad_f = lambda x: grad_reg(x.reshape(n, m), ker, blurred)
    return gradient_descend(f, n, m, grad_f, x0, max_iter, absolute_stop)


def regular_deblur_tot_var(blurred, n, m, ker, max_iter=50, absolute_stop=1.e-5):
    """
      Deblur an image using Gradient Descend method with total variation
      @param blurred: Blurred image
      @param n: Width
      @param m: Height
      @param ker: Gaussian kernel matrix
      @param max_iter: Maximum number of iterations
      @param absolute_stop: Absolute stop value
      @return: De-blurred image
    """
    x0 = np.zeros(n * m)
    x0[0] = 1

    f = lambda x: lstsq_totvar(x.reshape(n, m), ker, blurred, n, m, 0.01, 1.5)
    grad_f = lambda x: grad_lstsq_totvar(x.reshape(n, m), ker, blurred)

    return gradient_descend(f, n, m, grad_f, x0, max_iter, absolute_stop)


def elaborate_datasource(image_path, sigma, ker_len):
    """
    Given an image path first blur the image, then de-blur it using different methods
    @param image_path: Image path
    @param sigma: Blur operator
    @param ker_len: Gaussian kernel length
    """
    # Open the image as a matrix and show it
    picture = plt.imread(image_path)
    picture = picture[:, :, 0]
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

    # Phase 3.1: regular solution with conjugate gradient
    deblurred_gc = regular_deblur_cg(blurred, n, m, k)
    plt.imshow(deblurred_gc, cmap='gray')
    plt.show()

    # Phase 3.2: regular solution with gradient
    deblurred_gradient = regular_deblur_gradient(blurred, n, m, k)
    if deblurred_gradient is not None:
        plt.imshow(deblurred_gradient, cmap='gray')
        plt.show()

    # Phase 4: regular solution with gradient and total variation
    deblurred_totvar = regular_deblur_tot_var(blurred, n, m, k, max_iter=1)
    if deblurred_totvar is not None:
        plt.imshow(deblurred_totvar, cmap='gray')
        plt.show()


def main():
    ds1 = './datasource/six.png'

    elaborate_datasource(ds1, 0.5, 5)
    elaborate_datasource(ds1, 1, 7)
    elaborate_datasource(ds1, 1.3, 9)


if __name__ == "__main__":
    main()
