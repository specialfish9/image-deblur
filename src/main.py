import numpy as np


def lstsq(x, a, b):
    """
    least-squares function implementation
    """
    from helpers import A

    return (1 / 2) * np.linalg.norm(A(x, a) - b) ** 2


def lstsq_grad(x, a, b):
    """
    least-squares gradient implementation
    """
    from helpers import A, AT

    return AT(A(x, a), a) - AT(b, a)


def lstsq_tikhonov(x, a, b, lamb):
    """
    lstsq with Tikhonov regularization
    """
    return lstsq(x, a, b) + (1 / 2) * lamb * np.linalg.norm(x) ** 2


def lstsq_tikhonov_grad(x, a, b, lamb):
    """
    least-squares gradient with Tikhonov regularization
    """
    return lstsq_grad(x, a, b) + lamb * x


def total_variation(u, eps=1e-2):
    """
    Total Variation formula implementation
    """
    dx, dy = np.gradient(u)
    n2 = np.square(dx) + np.square(dy)

    return np.sqrt(n2 + eps ** 2).sum()


def tot_var_grad(u, eps=1e-2):
    """
    Total variation gradient
    """
    dx, dy = np.gradient(u)

    n2 = np.square(dx) + np.square(dy)
    den = np.sqrt(n2 + eps ** 2)

    fx = dx / den
    fy = dy / den

    d_fdx = np.gradient(fx, axis=0)
    d_fdy = np.gradient(fy, axis=1)

    div = (d_fdx + d_fdy)

    return -div


def lstsq_tot_var(x, a, b, lamb, eps=1e-2):
    """
    lstsq with total variation regularization
    """
    return lstsq(x, a, b) + lamb * total_variation(x, eps)


def lstsq_tot_var_grad(x, a, b, lamb, eps=1e-2):
    """
    lstsq gradient with total variation regularization
    """
    return lstsq_grad(x, a, b) + lamb * tot_var_grad(x, eps)


def blur_picture(picture, k, s=0.004):
    """
    Blur a picture using gaussian-blur and gaussian noise
    @param picture: Original picture
    @param k: Kernel matrix
    @param s: Standard deviation of gaussian noise
    @return: Blurred picture
    """
    from helpers import A

    # Generate noise
    noise = np.random.normal(size=picture.shape) * s

    # Blur and add noise
    return A(picture, k) + noise


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
    from scipy.optimize import minimize

    # Function to minimize
    f = lambda x: lstsq(x.reshape(n, m), k, blurred)
    # Gradient of f
    grad_f = lambda x: lstsq_grad(x.reshape(n, m), k, blurred).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    minimum = minimize(f, x0, jac=grad_f, options={'maxiter': max_iter}, method='CG')
    return minimum.x.reshape((n, m))


def regularized_deblur_cg(blurred, n, m, k, lamb, max_iter=50):
    """
    Deblur image finding the absolute minimum of lstsq  with Tikhonov regularization
    @param blurred: Image to de-blur
    @param n: Width
    @param m: Height
    @param k: Gaussian-kernel
    @param lamb: regularization parameter
    @param max_iter: Maximum number of iterations
    @return: De-blurred image
    """
    from scipy.optimize import minimize

    # Function to minimize
    f = lambda x: lstsq_tikhonov(x.reshape(n, m), k, blurred, lamb)
    # Gradient of f
    grad_f = lambda x: lstsq_tikhonov_grad(x.reshape(n, m), k, blurred, lamb).reshape(n * m)

    # Initial guess
    x0 = np.zeros(n * m)

    minimum = minimize(f, x0, jac=grad_f, options={'maxiter': max_iter}, method='CG')
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
    """
    Gradient descend method implementation.
    @param f: Function to minimize
    @param n: Width
    @param m: Height
    @param grad_f: Gradient of [f]
    @param x0: Initial guess
    @param max_iter: Maximum number of iterations
    @param absolute_stop: Absolute stop value
    @return: Min value calculated for [f]
    """
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


def regularized_deblur_gradient(blurred, n, m, ker, lamb, max_iter=50, absolute_stop=1.e-5):
    """
      Deblur an image using Gradient Descend method
      @param blurred: Blurred image
      @param n: Width
      @param m: Height
      @param ker: Gaussian kernel matrix
      @param max_iter: Maximum number of iterations
      @param absolute_stop: Absolute stop value
      @param lamb: Regularization parameter
      @return: De-blurred image
    """

    # initialize first values
    x0 = np.zeros(n * m)
    x0[0] = 1

    f = lambda x: lstsq_tikhonov(x.reshape(n, m), ker, blurred, lamb)
    grad_f = lambda x: lstsq_tikhonov_grad(x.reshape(n, m), ker, blurred, lamb)
    return gradient_descend(f, n, m, grad_f, x0, max_iter, absolute_stop)


def regularized_deblur_tot_var(blurred, n, m, ker, lamb, max_iter=50, absolute_stop=1.e-5):
    """
      Deblur an image using Gradient Descend method with total variation
      @param blurred: Blurred image
      @param n: Width
      @param m: Height
      @param ker: Gaussian kernel matrix
      @param max_iter: Maximum number of iterations
      @param absolute_stop: Absolute stop value
      @param lamb: regularization parameter
      @return: De-blurred image
    """
    x0 = np.zeros(n * m)
    x0[0] = 1

    f = lambda x: lstsq_tot_var(x.reshape(n, m), ker, blurred, lamb)
    grad_f = lambda x: lstsq_tot_var_grad(x.reshape(n, m), ker, blurred, lamb)

    return gradient_descend(f, n, m, grad_f, x0, max_iter, absolute_stop)


def print_res(original, deblurred, time):
    from skimage import metrics

    psnr = metrics.peak_signal_noise_ratio(original, deblurred)
    mse = metrics.mean_squared_error(original, deblurred)
    print('This is the mse: ', mse)
    print('This is the psnr: ', psnr)
    print('This is the time: ', time)


def elaborate_datasource(image_path, sigma, ker_len, noise_std_dev=5e-3, lamb=1e-5, show_pic=True):
    """
    Given an image path first blur the image, then de-blur it using different methods
    @param image_path: Image path
    @param sigma: Blur operator
    @param ker_len: Gaussian kernel length
    @param noise_std_dev: Standard deviation of noise
    @param lamb: Regularization parameter
    """
    from matplotlib import pyplot as plt
    from helpers import gaussian_kernel, psf_fft
    import time

    fig = plt.figure(figsize=(15, 10))
    rows = 2
    columns = 3

    # Open the image as a matrix and show it
    picture = plt.imread(image_path)
    picture = picture[:, :, 0]
    n, m = picture.shape

    if show_pic:
        fig.add_subplot(rows, columns, 1)
        plt.imshow(picture, cmap='gray')
        plt.axis("off")
        plt.title("Original")

    # Create the kernel
    kernel = gaussian_kernel(ker_len, sigma)
    k = psf_fft(kernel, ker_len, (n, m))

    # Phase 1: blur
    print("Phase 1")
    blurred = blur_picture(picture, k, noise_std_dev)

    if show_pic:
        fig.add_subplot(rows, columns, 2)
        plt.imshow(blurred, cmap='gray')
        plt.axis("off")
        plt.title("Blurred")

    start = time.time()

    # Phase 2: naive solution
    print("Phase 2")
    deblurred_naive = naive_deblur(blurred, n, m, k)
    t_2 = time.time()
    print_res(picture, deblurred_naive, t_2 - start)

    if show_pic:
        fig.add_subplot(rows, columns, 3)
        plt.imshow(deblurred_naive, cmap='gray')
        plt.axis("off")
        plt.title("Naive")

    # Phase 3.1: solution with conjugate gradient method and Tikhonov regularization
    print("Phase 3.1")
    deblurred_gc = regularized_deblur_cg(blurred, n, m, k, lamb)
    t_31 = time.time()
    print_res(picture, deblurred_gc, t_31 - t_2)

    if show_pic:
        fig.add_subplot(rows, columns, 4)
        plt.imshow(deblurred_gc, cmap='gray')
        plt.axis("off")
        plt.title("CG Tikhonov")

    # Phase 3.2: solution with gradient descend and Tikhonov regularization
    print("Phase 3.2")
    deblurred_gradient = regularized_deblur_gradient(blurred, n, m, k, lamb)
    t_32 = time.time()

    if deblurred_gradient is not None:
        print_res(picture, deblurred_gradient, t_32 - t_31)
        if show_pic:
            fig.add_subplot(rows, columns, 5)
            plt.imshow(deblurred_gradient, cmap='gray')
            plt.axis("off")
            plt.title("GD Tikhonov")

    # Phase 4: solution with gradient descend and total variation regularization
    print("Phase 4")
    deblurred_tot_var = regularized_deblur_tot_var(blurred, n, m, k, lamb)
    t_4 = time.time()

    if deblurred_tot_var is not None:
        print_res(picture, deblurred_tot_var, t_4 - t_32)
        if show_pic:
            fig.add_subplot(rows, columns, 6)
            plt.imshow(deblurred_tot_var, cmap='gray')
            plt.axis("off")
            plt.title("GD Tot Var")

    if show_pic:
        plt.show()


def main():
    ds1 = '../datasource/e6.webp'

    elaborate_datasource(ds1, 0.5, 5)
    elaborate_datasource(ds1, 1, 7)
    elaborate_datasource(ds1, 1.3, 9)


if __name__ == "__main__":
    main()
